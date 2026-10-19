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
    s = np.where(fin, blk, 0).sum(axis=(1, 3))
    n = fin.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan).astype(np.float32)


def detect(d: Dem, p: CandidateParams | None = None) -> dict:
    """Кандидаты: координаты центров в CRS DEM и матрица признаков N×len(FEATURE_NAMES)."""
    p = p or CandidateParams()
    k = max(1, int(round(p.work_res / d.res)))
    z = downsample_mean(d.dem, k)
    res = d.res * k
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


def to_features(c: dict, scores: np.ndarray, crs: str | None, region: str, p: CandidateParams,
                classes: tuple[str, ...] = ("kurgany_povrezhdennye", "kurgany_tselye"),
                source: str = "las") -> list[dict]:
    """Кандидаты → объекты пайплайна (круги в EPSG:3857, markup_type li).

    Кандидат выдаётся в обоих классах. Метрика считает классы независимо:
    в нужном классе это TP, в соседнем — FP с ценой 0,5. При N уверенностях
    у обоих классов одна уверенность; при N×3 (фон, повреждённый, целый) —
    у каждого своя. Лишнее отсекают пороги по классам (expds.metric sweep).
    """
    from shapely.geometry import Point

    per_class = scores[:, 1:1 + len(classes)] if scores.ndim == 2 else np.repeat(scores[:, None], len(classes), 1)
    out = []
    for x, y, s in zip(c["x"], c["y"], per_class):
        circle = Point(float(x), float(y)).buffer(p.circle_m, quad_segs=4)
        geom = to_crs(circle, crs, WEB_MERCATOR) if crs else circle
        for cls, conf in zip(classes, s):
            out.append({"geom": geom, "class_name": cls, "confidence": float(conf), "region_name": region,
                        "markup_type": "li", "source": source, "original_crs": crs})
    return out


def label(c: dict, crs: str, gt_items: list[dict], classes=("kurgany_povrezhdennye", "kurgany_tselye")) -> np.ndarray:
    """Метка кандидата: 1 + индекс класса эталонного кургана li, в котором его центр; 0 — фон.

    С одним классом в classes это обычная метка 0/1.
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    gts = [(g["geom"], classes.index(g["class_name"])) for g in gt_items
           if g["class_name"] in classes and g["markup_type"] == "li"]
    if not gts:
        return np.zeros(len(c["x"]), np.int8)
    tree = STRtree([g for g, _ in gts])
    y = np.zeros(len(c["x"]), np.int8)
    for k, (x0, y0) in enumerate(zip(c["x"], c["y"])):
        pt = to_crs(Point(float(x0), float(y0)), crs, WEB_MERCATOR)
        for j in tree.query(pt.buffer(2.5)):
            g, ci = gts[int(j)]
            if (g.geom_type == "Point" and g.distance(pt) <= 2.0) or g.contains(pt):
                y[k] = 1 + ci
                break
    return y


def collect(root: str | Path, gt_path: str | Path, p: CandidateParams | None = None) -> list[dict]:
    """Кандидаты и метки по всем LAS трейна: по записи на LAS (регион, семья, кандидаты, CRS, метки)."""
    from shapely.geometry import shape

    from .catalog import read_utm_json, scan_root
    from .lidar import build_dem
    from .naming import site_family

    p = p or CandidateParams()
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
            d = build_dem(las, crs=crs)
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
        booster = lgb.train(_lgb_params(multiclass), lgb.Dataset(x, y, feature_name=FEATURE_NAMES),
                            num_boost_round=rounds)
        for r in te:
            s = booster.predict(np.nan_to_num(r["c"]["features"], nan=-999.0)).astype(np.float32)
            feats += to_features(r["c"], s, r["crs"], r["region"], p, source=r["source"])
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
