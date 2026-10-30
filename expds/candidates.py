"""CPU-детектор курганов по рельефу (без видеокарты).

Курган — локальный холм. Алгоритм:
1. DEM приводится к рабочему шагу (по умолчанию 0,5 м) усреднением блоков;
2. локальный рельеф = высота − среднее окна (радиус ~15 м);
3. центры — локальные максимумы рельефа выше порога выраженности;
4. признаки кандидата — статистика диска и кольца вокруг центра;
5. оценка — эвристика по выраженности или модель LightGBM, если она обучена
   (`python -m expds.candidates train ...`); честная оценка до сабмита —
   `python -m expds.candidates oof ...` (обучение без своей «семьи» участков).

Это первый сабмит, страховка на случай таймаута и источник кандидатов для
нейросети. Выход — маленькие круги вокруг центров: метрика засчитывает
полигональный эталон, если внутренняя точка круга внутри кургана, а
покрытие меньшей фигуры ≥ 0,5 (круг меньше кургана → покрытие = 1).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import accel
from .crs import WEB_MERCATOR, to_crs
from .lidar import Dem

log = logging.getLogger(__name__)

FEATURE_NAMES = ["relief", "disk", "ring", "ring_std", "prominence", "ring_min", "ring_max", "contrast"]


@dataclass
class CandidateParams:
    """Параметры детектора (метры). Подбираются на трейне по официальной метрике."""
    work_res: float = 0.5
    relief_radius_m: float = 15.0
    peak_radius_m: float = 4.0
    min_relief_m: float = 0.12
    r_in_m: float = 2.5
    r_out_m: float = 8.0
    circle_m: float = 2.0
    max_candidates: int = 50_000


def downsample_mean(dem: np.ndarray, k: int) -> np.ndarray:
    """Усреднение блоков k×k без учёта NaN (k ≥ 1)."""
    if k <= 1:
        return dem.astype(np.float32)
    h, w = dem.shape
    hh, ww = h // k * k, w // k * k
    blk = dem[:hh, :ww].reshape(hh // k, k, ww // k, k)
    fin = np.isfinite(blk)
    s = np.where(fin, blk, 0).astype(np.float64).sum(axis=(1, 3))
    n = fin.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan).astype(np.float32)


def to_work_grid(dem: np.ndarray, src_res: float, dst_res: float) -> tuple[np.ndarray, float]:
    """DEM к рабочему шагу. Целый множитель совпадает с обучением (блоки k×k).

    Крупный LAS не влезает в сетку 0,1 м, и шаг DEM становится 0,2–0,3 м. Округление
    0,5/0,3 до 2 даёт шаг 0,6 м вместо 0,5 м, на котором училась модель.
    Считается в Rust (`expds_core.work_grid`), NumPy ниже — запасной путь.
    """
    dem = np.ascontiguousarray(dem, np.float32)
    if accel._rs is not None and hasattr(accel._rs, "work_grid"):
        arr, step = accel._rs.work_grid(dem, float(src_res), float(dst_res))
        return arr, float(step)
    if src_res <= 0 or dst_res <= 0:
        raise ValueError(f"шаг сетки должен быть > 0, получено {src_res}, {dst_res}")
    ratio = dst_res / src_res
    if ratio <= 1.02:
        return dem.astype(np.float32), float(src_res)
    k = int(round(ratio))
    if k >= 1 and abs(src_res * k - dst_res) <= 0.02 * dst_res:
        return downsample_mean(dem, k), float(src_res * k)
    h, w = dem.shape
    nh = max(1, int(np.floor(h * src_res / dst_res)))
    nw = max(1, int(np.floor(w * src_res / dst_res)))
    cols = np.minimum((np.arange(w) * src_res / dst_res).astype(np.int64), nw - 1)
    sums = np.zeros(nh * nw, np.float64)
    cnts = np.zeros(nh * nw, np.float64)
    # Полосами: целая индексная карта h×w на крупном LAS — это сотни мегабайт и падение процесса.
    for r0 in range(0, h, 256):
        r1 = min(h, r0 + 256)
        rows = np.minimum((np.arange(r0, r1) * src_res / dst_res).astype(np.int64), nh - 1)
        idx = (rows[:, None] * nw + cols[None, :]).ravel()
        block = dem[r0:r1]
        fin = np.isfinite(block)
        sums += np.bincount(idx, weights=np.where(fin, block, 0.0).ravel(), minlength=nh * nw)
        cnts += np.bincount(idx, weights=fin.ravel().astype(np.float64), minlength=nh * nw)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return out.reshape(nh, nw).astype(np.float32), float(dst_res)


def detect(d: Dem, p: CandidateParams | None = None) -> dict:
    """Кандидаты: координаты центров в CRS DEM и матрица признаков N×len(FEATURE_NAMES)."""
    p = p or CandidateParams()
    z, res = to_work_grid(d.dem, d.res, p.work_res)
    relief = accel.local_relief(z, max(1, int(round(p.relief_radius_m / res))))
    relief = np.where(np.isfinite(z), relief, np.nan).astype(np.float32)
    rows, cols, vals = accel.find_peaks(relief, max(1, int(round(p.peak_radius_m / res))),
                                        p.min_relief_m, p.max_candidates)
    st = accel.ring_stats(z, rows, cols, p.r_in_m / res, p.r_out_m / res)
    contrast = st[:, 3] / (st[:, 2] + 0.05)
    feats = np.column_stack([vals, st, contrast]).astype(np.float32)
    xs = d.left + (cols + 0.5) * res
    ys = d.top - (rows + 0.5) * res
    return {"x": xs, "y": ys, "features": feats, "res": res}


def heuristic_score(feats: np.ndarray) -> np.ndarray:
    """Оценка 0..1 без обучения: логистическая функция выраженности и контраста."""
    prom = np.nan_to_num(feats[:, 4], nan=0.0)
    con = np.nan_to_num(feats[:, 7], nan=0.0)
    return (1.0 / (1.0 + np.exp(-(prom - 0.35) / 0.12)) * (1.0 / (1.0 + np.exp(-(con - 2.0))))).astype(np.float32)


def dome_multiplier(d: Dem, found: dict, near_m: float = 5.0, far_m: float = 40.0) -> np.ndarray:
    """Множитель 0,9…1,1 по форме насыпи. Пустой список кандидатов — пустой множитель."""
    from .dome import priors

    z, step = to_work_grid(d.dem, d.res, float(found["res"]))
    return priors(z, float(step), found["x"], found["y"], d.left, d.top, near_m, far_m)


def score(feats: np.ndarray, model_path: str | Path | None = None) -> np.ndarray:
    """Оценка кандидатов: LightGBM, если модель есть, иначе эвристика.

    Двухклассовая модель даёт N уверенностей «курган»; трёхклассовая —
    N×3 вероятностей (фон, повреждённый, целый).
    """
    if model_path and Path(model_path).exists():
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_path))
        return booster.predict(np.nan_to_num(feats, nan=-999.0)).astype(np.float32)
    return heuristic_score(feats)


def add_prob_peaks(c: dict, prob: np.ndarray, transform, min_prob: float = 0.3, radius_m: float = 2.5,
                   dedupe_m: float = 2.0) -> dict:
    """Кандидаты из пиков вероятности модели там, где пика рельефа ближе dedupe_m нет.

    Пики рельефа пропускают мелкие и низкие курганы (до 30 м² — половину), а модель их видит.
    Признаков рельефа у таких кандидатов нет (NaN): в режиме rescore оценка — вероятность модели.
    """
    from scipy.spatial import cKDTree

    res = abs(transform.a)
    rows, cols, _ = accel.find_peaks(np.nan_to_num(prob, nan=0.0).astype(np.float32),
                                     max(1, round(radius_m / res)), min_prob)
    if rows.size == 0:
        return c
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e
    if len(c["x"]):
        dist, _ = cKDTree(np.column_stack([c["x"], c["y"]])).query(np.column_stack([x, y]))
        x, y = x[dist > dedupe_m], y[dist > dedupe_m]
    nf = c["features"].shape[1]
    return {**c, "x": np.r_[c["x"], x], "y": np.r_[c["y"], y],
            "features": np.vstack([c["features"], np.full((len(x), nf), np.nan, np.float32)])}


def sample_probs(c: dict, probs: np.ndarray, transform, window_px: int = 5) -> np.ndarray:
    """Вероятности модели в центрах кандидатов: максимум в окне window_px² (K×H×W → N×K)."""
    if len(c["x"]) == 0:
        return np.zeros((0, probs.shape[0]), np.float32)
    col = np.clip(np.floor((c["x"] - transform.c) / transform.a).astype(np.int64), 0, probs.shape[2] - 1)
    row = np.clip(np.floor((transform.f - c["y"]) / transform.a).astype(np.int64), 0, probs.shape[1] - 1)
    probs = np.ascontiguousarray(probs, np.float32)
    if accel._rs is not None and hasattr(accel._rs, "sample_window_max"):
        return accel._rs.sample_window_max(probs, row, col, int(window_px))
    from scipy.ndimage import maximum_filter

    return np.stack([maximum_filter(p, size=window_px)[row, col] for p in probs], 1).astype(np.float32)


def to_features(c: dict, scores: np.ndarray, crs: str | None, region: str, p: CandidateParams,
                classes: tuple[str, ...] = ("kurgany_povrezhdennye", "kurgany_tselye"),
                source: str = "las", markup: str = "li") -> list[dict]:
    """Кандидаты → объекты пайплайна (круги в EPSG:3857).

    Кандидат выдаётся в обоих классах. Метрика считает классы независимо:
    в нужном классе это TP, в соседнем — FP с ценой 0,5. При N уверенностях
    у обоих классов одна уверенность; при N×3 (фон, повреждённый, целый) —
    у каждого своя. Лишнее отсекают пороги по классам (expds.metric sweep).
    """
    from shapely import buffer, points

    xs = np.asarray(c["x"], np.float64)
    ys = np.asarray(c["y"], np.float64)
    if xs.size == 0:
        return []
    # Один вызов GEOS и один пересчёт CRS на все вершины, а не по кругу.
    geoms = buffer(points(xs, ys), float(p.circle_m), quad_segs=4)
    if crs and crs != WEB_MERCATOR:
        geoms = to_crs(geoms, crs, WEB_MERCATOR)
    per_class = scores[:, 1:1 + len(classes)] if scores.ndim == 2 else np.repeat(scores[:, None], len(classes), 1)
    out = []
    for geom, s in zip(geoms, per_class):
        for cls, conf in zip(classes, s):
            out.append({"geom": geom, "class_name": cls, "confidence": float(conf), "region_name": region,
                        "markup_type": markup, "source": source, "original_crs": crs})
    return out


def label(c: dict, crs: str, gt_items: list[dict], classes=("kurgany_povrezhdennye", "kurgany_tselye"),
          markup: str = "li") -> np.ndarray:
    """Метка кандидата: 1 + индекс класса эталона модальности markup, в котором его центр; 0 — фон.

    С одним классом в classes это обычная метка 0/1. Эталонные точки засчитываются
    ближе 2 м в метрах CRS кандидатов (UTM), как в метрике, а не в единицах EPSG:3857.
    """
    import shapely
    from shapely.strtree import STRtree

    gts = [(g["geom"], classes.index(g["class_name"])) for g in gt_items
           if g["class_name"] in classes and g["markup_type"] == markup]
    y = np.zeros(len(c["x"]), np.int8)
    if not gts or len(c["x"]) == 0:
        return y
    pts = shapely.points(np.asarray(c["x"]), np.asarray(c["y"]))
    geoms = [to_crs(g, WEB_MERCATOR, crs) if crs != WEB_MERCATOR else g for g, _ in gts]
    geoms = [g.buffer(2.0) if g.geom_type == "Point" else g for g in geoms]
    hit_pt, hit_gt = STRtree(geoms).query(pts, predicate="intersects")
    order = np.argsort(hit_gt, kind="stable")          # при нескольких эталонах — первый по порядку
    for k, j in zip(hit_pt[order][::-1], hit_gt[order][::-1]):
        y[k] = 1 + gts[int(j)][1]
    return y


def collect(root: str | Path, gt_path: str | Path, p: CandidateParams | None = None,
            dem: dict | None = None) -> list[dict]:
    """Кандидаты и метки по всем LAS трейна: по записи на LAS (регион, семья, кандидаты, CRS, метки).

    dem — {res, fill} DEM, как в пайплайне без модели li (по умолчанию lidar.MODEL_DEM).
    """
    from shapely.geometry import shape

    from .catalog import read_utm_json, scan_root
    from .lidar import MODEL_DEM, build_dem
    from .naming import site_family

    p = p or CandidateParams()
    dem = dem or MODEL_DEM
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))["features"]
    gt_by_region: dict[str, list[dict]] = {}
    for f in gt:
        pr = f["properties"]
        gt_by_region.setdefault(pr["region_name"], []).append(
            {"geom": shape(f["geometry"]), "class_name": pr["class_name"], "markup_type": pr["markup_type"]})
    out = []
    for reg in scan_root(root):
        crs = read_utm_json(reg.utm_json)
        for las in reg.las:
            d = build_dem(las, res=dem["res"], fill=dem["fill"], crs=crs)
            c = detect(d, p)
            if len(c["x"]) == 0 or not d.crs:
                continue
            y = label(c, d.crs, gt_by_region.get(reg.name, []))
            out.append({"region": reg.name, "family": site_family(reg.name), "source": las.name,
                        "c": c, "crs": d.crs, "y": y})
            log.info("%s: кандидатов %d, позитивов %d", reg.name, len(y), int((y > 0).sum()))
    return out


def _lgb_params(multiclass: bool) -> dict:
    base = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.9,
            "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}
    return {**base, "objective": "multiclass", "num_class": 3} if multiclass else {**base, "objective": "binary"}


def family_folds(families: list[str], weights: list[int], k: int) -> dict[str, int]:
    """Семьи по k фолдам так, чтобы позитивы распределились ровно (крупные — первыми)."""
    tot: dict[str, int] = {}
    for f, w in zip(families, weights):
        tot[f] = tot.get(f, 0) + w
    load = [0] * k
    fold = {}
    for f in sorted(tot, key=lambda f: (-tot[f], f)):
        i = int(np.argmin(load))
        fold[f] = i
        load[i] += tot[f]
    return fold


def oof_predictions(recs: list[dict], p: CandidateParams, folds: int = 5, multiclass: bool = True,
                    rounds: int = 300) -> list[dict]:
    """Предсказания «вне обучения»: оценщик каждой семьи учится только на других семьях."""
    import lightgbm as lgb

    fold = family_folds([r["family"] for r in recs], [int((r["y"] > 0).sum()) for r in recs], folds)
    feats: list[dict] = []
    for k in sorted(set(fold.values())):
        tr = [r for r in recs if fold[r["family"]] != k]
        te = [r for r in recs if fold[r["family"]] == k]
        if not tr or not te:
            continue
        x = np.nan_to_num(np.vstack([r["c"]["features"] for r in tr]), nan=-999.0)
        y = np.concatenate([r["y"] for r in tr])
        y = y if multiclass else (y > 0).astype(np.int8)
        names = FEATURE_NAMES if x.shape[1] == len(FEATURE_NAMES) else "auto"
        booster = lgb.train(_lgb_params(multiclass), lgb.Dataset(x, y, feature_name=names), num_boost_round=rounds)
        for r in te:
            s = booster.predict(np.nan_to_num(r["c"]["features"], nan=-999.0)).astype(np.float32)
            feats += to_features(r["c"], s, r["crs"], r["region"], p, source=r["source"],
                                 markup=r.get("markup", "li"))
        log.info("фолд %d: обучение %d LAS, проверка %d LAS", k, len(tr), len(te))
    return feats


def _cli() -> None:
    ap = argparse.ArgumentParser(description="CPU-кандидаты курганов: обучение оценщика LightGBM")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="собрать признаки по LAS трейна и обучить LightGBM")
    t.add_argument("--root", required=True)
    t.add_argument("--gt", required=True, help="эталон из expds.gt")
    t.add_argument("--out", required=True, help="файл модели, например models/kurgan_lgbm.txt")
    t.add_argument("--val-families", nargs="*", default=[], help="«семьи» участков для проверки")
    t.add_argument("--multiclass", action="store_true", help="три класса: фон, повреждённый, целый")
    o = sub.add_parser("oof", help="предсказания вне обучения по «семьям» для оценки и подбора порогов")
    o.add_argument("--root", required=True)
    o.add_argument("--gt", required=True, help="эталон из expds.gt")
    o.add_argument("--out", required=True, help="GeoJSON предсказаний (низкий порог, для expds.metric sweep)")
    o.add_argument("--folds", type=int, default=5)
    o.add_argument("--binary", action="store_true", help="одна уверенность на оба класса")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import lightgbm as lgb

    p = CandidateParams()
    recs = collect(args.root, args.gt, p)
    if args.cmd == "oof":
        from . import post, writer

        feats = oof_predictions(recs, p, args.folds, multiclass=not args.binary)
        feats = post.finalize(post.merge_within_markup(feats))
        writer.write_geojson(feats, args.out)
        print(json.dumps({"las": len(recs), "families": len({r["family"] for r in recs}),
                          "positives": int(sum((r["y"] > 0).sum() for r in recs)), "written": len(feats),
                          "out": args.out}, ensure_ascii=False))
        return
    X = np.nan_to_num(np.vstack([r["c"]["features"] for r in recs]), nan=-999.0)
    Y = np.concatenate([r["y"] for r in recs])
    Y = Y if args.multiclass else (Y > 0).astype(np.int8)
    fam = np.concatenate([[r["family"]] * len(r["y"]) for r in recs])
    is_val = np.isin(fam, args.val_families) if args.val_families else np.zeros(len(Y), bool)
    dtrain = lgb.Dataset(X[~is_val], Y[~is_val], feature_name=FEATURE_NAMES)
    valid = [lgb.Dataset(X[is_val], Y[is_val], reference=dtrain)] if is_val.any() else []
    booster = lgb.train(_lgb_params(args.multiclass), dtrain, num_boost_round=400, valid_sets=valid,
                        callbacks=[lgb.early_stopping(40)] if valid else None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.out)
    print(json.dumps({"train": int((~is_val).sum()), "val": int(is_val.sum()), "positives": int((Y > 0).sum()),
                      "params": asdict(p), "model": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
