"""Оценка официальной метрикой и точный подбор порогов по классам.

Официальный compute_metrics.py в архив не кладётся: путь к вашей копии с
платформы передаётся явно (--metric) или через EXPDS_METRIC. Свою метрику
не пишем: геометрические проверки берутся из официального кода.

Почему подбор точный и быстрый. Метрика сопоставляет предсказания жадно, по
убыванию confidence, и каждый класс считает отдельно. Отбросить предсказания
ниже порога τ — значит отрезать «хвост» отсортированного списка; решения для
более уверенных предсказаний от хвоста не зависят. Поэтому достаточно ОДНОГО
прохода сопоставления на всех кандидатах: запоминаем, какие попали (hit), и для
любого τ получаем TP = hit с conf ≥ τ, FP = (всего с conf ≥ τ) − TP, FN = GT − TP.
Перебираются все различимые пороги, а не сетка; совпадение с полным официальным
расчётом проверяет verify() (и тест test_metric_exact.py).

    python -m expds.metric score --pred out.geojson --gt work/gt_val.geojson --metric C:/koz3/compute_metrics.py
    python -m expds.metric sweep --pred cand.geojson --gt work/gt_train.geojson --metric ... \
        --markup li --regions holdout.txt --bootstrap 300 --out configs/thresholds.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import CLASS_WEIGHTS, CLASSES

EPS = 1e-6


@lru_cache(maxsize=2)
def load_official(path: str | None = None):
    """Импортирует официальный compute_metrics.py как модуль (рядом может лежать его config.py)."""
    path = path or os.environ.get("EXPDS_METRIC")
    if not path or not Path(path).exists():
        raise FileNotFoundError("укажите путь к официальному compute_metrics.py (--metric или EXPDS_METRIC)")
    parent = str(Path(path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("koz3_compute_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def score(pred: dict, gt: dict, metric_path: str | None = None) -> dict:
    """Полный результат официальной метрики (final_score, per_class, auxiliary)."""
    mod = load_official(metric_path)
    return mod.F1Calculator().calculate_f1(pred, gt)


# ─────────────────────────── один проход сопоставления ───────────────────────

@dataclass
class ClassCurve:
    """Итог одного прохода для класса: уверенности по убыванию и попадания."""
    conf: np.ndarray
    hit: np.ndarray
    pred_region: list[str]
    n_gt: int
    gt_regions: Counter = field(default_factory=Counter)
    pred_group: list[str] = field(default_factory=list)


def threshold_group(props: dict) -> str:
    """Группа порога объекта, как в post.apply_thresholds: threshold_group, иначе markup_type."""
    return str(props.get("threshold_group") or props.get("markup_type") or "")


_UTM_GEOM_CACHE: dict[tuple, dict[int, tuple[object, object]]] = {}
_SHM_CACHE_DIR = Path("/dev/shm/expds_utm_cache")


def _todo_batch_hash(todo_feats: list, todo_meta: list, mode: str, is_gt: bool, r_tol: float, c_pts: int) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(f"{mode}:{is_gt}:{r_tol}:{c_pts}:{len(todo_feats)}:".encode())
    confs = np.array([m[1] for m in todo_meta], dtype=np.float64)
    h.update(confs.tobytes())
    step = max(1, len(todo_feats) // 512)
    for i in range(0, len(todo_feats), step):
        _, gdict, _ = todo_feats[i]
        cls, _, reg, grp, _ = todo_meta[i]
        h.update(f"{cls}:{reg}:{grp}:{gdict.get('type')}:{gdict.get('coordinates')!r};".encode())
    return h.hexdigest()


def _collect(mod, calc, fc: dict, is_gt: bool) -> dict[str, list[tuple]]:
    """Повторяет F1Calculator._collect_by_class, сохраняя регион объекта (с кэшем в RAM + /dev/shm и векторизацией UTM)."""
    import pickle
    import shapely

    out: dict[str, list[tuple]] = {}
    mode, r_tol, c_pts = calc.input_crs_mode, calc.r_tolerance, calc.circle_points
    sub_cache = _UTM_GEOM_CACHE.setdefault((mode, is_gt, r_tol, c_pts), {})
    if len(sub_cache) > 1_200_000:
        sub_cache.clear()
    target_set = set(calc.target_classes)
    todo_feats: list[tuple[dict, object, int]] = []
    todo_meta: list[tuple[str, float, str, str, int]] = []
    ordered_slots: list[tuple[str, object, float, str, str]] = []

    for feature in fc.get("features", []):
        props = feature.get("properties", {}) or {}
        cls = mod.normalize_class_name(props.get("class_name", "unknown"))
        if cls is None or cls not in target_set:
            continue
        gdict = feature.get("geometry")
        if not gdict:
            continue
        conf = mod._get_confidence(feature)
        reg = str(props.get("region_name", "default_source"))
        grp = threshold_group(props)
        gid = id(gdict)
        hit = sub_cache.get(gid)
        if hit is not None and hit[0] is gdict:
            ordered_slots.append((cls, hit[1], conf, reg, grp))
        else:
            slot_idx = len(ordered_slots)
            ordered_slots.append((cls, None, conf, reg, grp))
            todo_feats.append((feature, gdict, gid))
            todo_meta.append((cls, conf, reg, grp, slot_idx))

    if todo_feats:
        shm_loaded = False
        shm_file = None
        if len(todo_feats) >= 200 and Path("/dev/shm").is_dir():
            try:
                bhash = _todo_batch_hash(todo_feats, todo_meta, mode, is_gt, r_tol, c_pts)
                shm_file = _SHM_CACHE_DIR / f"{bhash}.wkb.pkl"
                if shm_file.exists():
                    valid_idx, wkbs = pickle.loads(shm_file.read_bytes())
                    geoms = shapely.from_wkb(wkbs)
                    for g, idx in zip(geoms, valid_idx):
                        _, gdict, gid = todo_feats[idx]
                        sub_cache[gid] = (gdict, g)
                        cls, conf, reg, grp, slot_idx = todo_meta[idx]
                        ordered_slots[slot_idx] = (cls, g, conf, reg, grp)
                    shm_loaded = True
            except Exception:
                shm_loaded = False

        if not shm_loaded:
            raw_geoms = []
            valid_idx = []
            for idx, (feature, gdict, gid) in enumerate(todo_feats):
                g = mod._extract_geometry(feature)
                if g is not None:
                    raw_geoms.append(g)
                    valid_idx.append(idx)
            if raw_geoms:
                if mode == "utm":
                    proj_geoms = raw_geoms
                else:
                    proj_geoms = list(raw_geoms)
                    need_idx = [
                        k_i for k_i, g in enumerate(raw_geoms)
                        if mode == "webmercator" or mod._geom_max_abs_x(g) > mod._UTM_ABS_X_THRESHOLD
                    ]
                    if need_idx:
                        try:
                            sub_raw = [raw_geoms[k_i] for k_i in need_idx]
                            cents = shapely.centroid(sub_raw)
                            xs = shapely.get_x(cents)
                            ys = shapely.get_y(cents)
                            lons, lats = mod._get_transformer_3857_to_wgs84().transform(xs, ys)
                            zones = np.clip(((np.asarray(lons) + 180.0) // 6).astype(np.int32) + 1, 1, 60)
                            epsgs = np.where(np.asarray(lats) >= 0, 32600, 32700) + zones
                            for epsg in np.unique(epsgs):
                                z_mask = np.nonzero(epsgs == epsg)[0]
                                tr = mod._get_transformer_3857_to_utm(int(epsg))
                                z_geoms = [sub_raw[z_i] for z_i in z_mask]
                                z_proj = shapely.transform(
                                    z_geoms, lambda c, _tr=tr: np.column_stack(_tr.transform(c[:, 0], c[:, 1]))
                                )
                                for z_i, pg in zip(z_mask, z_proj):
                                    proj_geoms[need_idx[z_i]] = pg
                        except Exception:
                            for k_i in need_idx:
                                proj_geoms[k_i] = mod._to_utm(raw_geoms[k_i], mode)
                final_geoms = []
                for g, idx in zip(proj_geoms, valid_idx):
                    if g.geom_type == "Point" and not is_gt:
                        g = mod._point_to_circle(g, r_tol, c_pts)
                    final_geoms.append(g)
                    _, gdict, gid = todo_feats[idx]
                    sub_cache[gid] = (gdict, g)
                    cls, conf, reg, grp, slot_idx = todo_meta[idx]
                    ordered_slots[slot_idx] = (cls, g, conf, reg, grp)
                if shm_file is not None and final_geoms:
                    try:
                        _SHM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        wkbs = shapely.to_wkb(final_geoms)
                        tmp = shm_file.with_suffix(f".tmp.{os.getpid()}")
                        tmp.write_bytes(pickle.dumps((valid_idx, wkbs), protocol=pickle.HIGHEST_PROTOCOL))
                        tmp.replace(shm_file)
                    except Exception:
                        pass

    for cls, g, conf, reg, grp in ordered_slots:
        if g is not None:
            out.setdefault(cls, []).append((g, conf, reg, grp))
    return out


def build_curves(pred: dict, gt: dict, metric_path: str | None = None) -> dict[str, ClassCurve]:
    """Официальное жадное сопоставление по каждому классу, один раз на всех кандидатах.

    Кандидаты на пару ищутся пространственным индексом; порядок перебора эталонов
    (по возрастанию индекса) и правило выбора лучшего — как в официальном коде.
    """
    from shapely.strtree import STRtree

    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    return match_collected(calc, _collect(mod, calc, pred, False), _collect(mod, calc, gt, True))


def match_collected(calc, preds: dict, gts: dict) -> dict[str, ClassCurve]:
    """Сопоставление по уже собранным (геометрия UTM, conf, участок, группа) спискам.

    Вынесено из build_curves, чтобы большой стек можно было собирать кусками
    (по семьям, пачками) и не держать весь GeoJSON в памяти.
    """
    from shapely.strtree import STRtree

    curves: dict[str, ClassCurve] = {}
    for cls in calc.target_classes:
        p, g = preds.get(cls, []), gts.get(cls, [])
        order = sorted(range(len(p)), key=lambda i: p[i][1], reverse=True)   # стабильно, как sorted()
        hits = np.zeros(len(p), dtype=bool)
        if p and g:
            gg = [x[0] for x in g]
            tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gg])
            pg_ord = [p[i][0] for i in order]
            q_pred, q_gt = tree.query(pg_ord)
            if len(q_pred):
                splits = np.flatnonzero(np.diff(q_pred)) + 1
                rank_groups = np.split(q_gt, splits)
                unique_ranks = q_pred[np.r_[0, splits]]
                matched: set[int] = set()
                for rank, cand_js in zip(unique_ranks, rank_groups):
                    r_int = int(rank)
                    pg = pg_ord[r_int]
                    if pg.geom_type != "Polygon":
                        continue
                    best, best_j = float("-inf"), -1
                    for j in sorted(int(k) for k in cand_js):
                        if j in matched:
                            continue
                        ok, s = calc._match_quality(pg, gg[j])
                        if ok and s > best + 1e-9:
                            best, best_j = s, j
                    if best_j >= 0:
                        matched.add(best_j)
                        hits[r_int] = True
        curves[cls] = ClassCurve(conf=np.array([p[i][1] for i in order], dtype=np.float64), hit=hits,
                                 pred_region=[p[i][2] for i in order], n_gt=len(g),
                                 gt_regions=Counter(x[2] for x in g), pred_group=[p[i][3] for i in order])
    return curves


def _group_cuts(conf: np.ndarray, hit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Допустимые срезы группы (уверенности по убыванию): сколько оставить, TP и порог на каждом."""
    if conf.size == 0:
        return np.array([0]), np.array([0.0]), np.array([np.inf])
    last = np.r_[np.flatnonzero(conf[1:] != conf[:-1]), conf.size - 1]
    keep = np.r_[0, last + 1]
    tp = np.r_[0.0, np.cumsum(hit)[last].astype(float)]
    lower = np.r_[conf[last[:-1] + 1], -np.inf] if last.size else np.array([-np.inf])
    taus = np.r_[conf[0] + 1e-6, np.where(np.isfinite(lower), (conf[last] + lower) / 2, 0.0)]
    return keep, tp, taus


def _f1_along(groups: list[str], cuts: dict, k: dict, g: str, n_gt: int) -> np.ndarray:
    """F1 класса, когда у группы g меняется срез, а остальные группы зафиксированы."""
    tp_o = sum(cuts[h][1][k[h]] for h in groups if h != g)
    n_o = sum(cuts[h][0][k[h]] for h in groups if h != g)
    keep, tp, _ = cuts[g]
    t, n = tp_o + tp, n_o + keep
    return np.where(n > 0, _f1(t, n - t, n_gt - t), 0.0)


def _plateau_indices(groups: list[str], cuts: dict, k: dict, n_gt: int) -> dict:
    """Один проход: каждая группа встаёт на середину своего плато при уже найденных остальных.

    Повторный подъём здесь не нужен. Середина чуть ниже острого максимума, и следующий
    проход снова уехал бы к краю.
    """
    chosen = dict(k)
    for g in groups:
        chosen[g] = plateau_index(_f1_along(groups, cuts, chosen, g, n_gt))
    return chosen


def sweep_groups(curves: dict[str, ClassCurve], rounds: int = 20, pick: str = "max") -> dict:
    """Пороги по (группа, класс), максимизирующие общий F1 класса по всем группам.

    Метрика сводит класс по всем модальностям и участкам: F1 ≈ 2TP / (TP + 0,5·FP + GT).
    Покоординатный подъём: порог одной группы — лучший при остальных. Два старта
    (все группы выключены, все включены), берётся лучший. Попадания — из общего
    прохода, поэтому группы, борющиеся за один эталон, оцениваются приближённо;
    точный балл — verify_groups.

    pick="max" оставляет острый максимум. pick="plateau" после него сдвигает каждую
    группу на середину плато F1 (допуск 0,002). В конфиг этот сдвиг не записывается,
    пока официальный балл на новых участках его не подтвердит.
    """
    out: dict[str, dict] = {}
    for cls, c in curves.items():
        groups = sorted(set(c.pred_group)) if c.pred_group else []
        if c.n_gt == 0:
            continue
        if not groups:
            out[cls] = {"f1": 0.0, "n_gt": c.n_gt, "groups": {}}
            continue
        g_arr = np.array(c.pred_group)
        cuts = {g: _group_cuts(c.conf[g_arr == g], c.hit[g_arr == g]) for g in groups}

        def f1_of(k: dict) -> float:
            tp = sum(cuts[g][1][k[g]] for g in groups)
            n = sum(cuts[g][0][k[g]] for g in groups)
            return float(_f1(tp, n - tp, c.n_gt - tp)) if n else 0.0

        best_k, best_f = None, -1.0
        for start in ("none", "all"):
            k = {g: 0 if start == "none" else len(cuts[g][0]) - 1 for g in groups}
            for _ in range(rounds):
                changed = False
                for g in groups:
                    f = _f1_along(groups, cuts, k, g, c.n_gt)
                    j = int(np.argmax(f))
                    if j != k[g] and f[j] > f[k[g]] + 1e-12:
                        k[g], changed = j, True
                if not changed:
                    break
            f = f1_of(k)
            if f > best_f:
                best_f, best_k = f, dict(k)
        if pick == "plateau" and best_k is not None:
            best_k = _plateau_indices(groups, cuts, best_k, c.n_gt)
            best_f = f1_of(best_k)
        elif pick != "max":
            raise ValueError(f"неизвестный выбор порога: {pick}")
        out[cls] = {"f1": best_f, "n_gt": c.n_gt,
                    "groups": {g: {"threshold": round(float(cuts[g][2][best_k[g]]), 6),
                                   "kept": int(cuts[g][0][best_k[g]]), "tp": int(cuts[g][1][best_k[g]])}
                               for g in groups}}
    return out


def official_weighted(per_class: dict[str, dict]) -> float:
    """Взвешенный балл по F1 классов с эталоном (как в метрике)."""
    num = sum(CLASS_WEIGHTS.get(c, 0.0) * v["f1"] for c, v in per_class.items() if v.get("n_gt", 0) > 0)
    den = sum(CLASS_WEIGHTS.get(c, 0.0) for c, v in per_class.items() if v.get("n_gt", 0) > 0)
    return num / den if den else 0.0


def apply_group_thresholds(fc: dict, taus: dict[str, dict[str, float]], metric_path: str | None = None) -> dict:
    """Оставляет объекты с confidence ≥ taus[группа][класс]; нет порога — объект остаётся."""
    norm = load_official(metric_path).normalize_class_name
    feats = []
    for f in fc.get("features", []):
        pr = f.get("properties") or {}
        tau = taus.get(threshold_group(pr), {}).get(norm(pr.get("class_name")), 0.0)
        if float(pr.get("confidence", 1.0)) >= tau:
            feats.append(f)
    return {**{k: v for k, v in fc.items() if k != "features"}, "features": feats}


def _f1(tp, fp, fn):
    """F1 ровно как в метрике: FP с весом 0,5 в точности, эпсилон 1e-6."""
    precision = tp / (tp + 0.5 * fp + EPS)
    recall = tp / (tp + fn + EPS)
    return 2 * precision * recall / (precision + recall + EPS)


def curve_table(c: ClassCurve) -> tuple[np.ndarray, np.ndarray]:
    """Все различимые пороги класса (оставляем conf ≥ τ) и F1 на каждом; первый — «ничего не выдавать»."""
    if c.conf.size == 0:
        return np.array([np.inf]), np.array([0.0])
    last = np.r_[np.flatnonzero(c.conf[1:] != c.conf[:-1]), c.conf.size - 1]
    tp = np.cumsum(c.hit)[last].astype(float)
    n = (last + 1).astype(float)
    return np.r_[np.inf, c.conf[last]], np.r_[0.0, _f1(tp, n - tp, c.n_gt - tp)]


def plateau_index(f1: np.ndarray, tol: float = 0.002) -> int:
    """Середина плато: все пороги, где F1 не ниже максимума больше чем на tol.

    Острый оптимум на одном значении confidence плохо переносится на новые участки.
    При ровной кривой берётся середина, а не левый или правый край.
    """
    scores = np.asarray(f1, dtype=float)
    if scores.size == 0:
        return 0
    ok = np.flatnonzero(scores >= float(scores.max()) - tol)
    return int(ok[len(ok) // 2])


def score_at(curves: dict[str, ClassCurve], thresholds: dict[str, float]) -> tuple[float, dict]:
    """Итоговый балл при порогах (классы без эталона из среднего исключаются, как в метрике)."""
    num = den = 0.0
    per = {}
    for cls, c in curves.items():
        tau = thresholds.get(cls) or 0.0
        kept = c.conf >= tau
        tp = int(c.hit[kept].sum())
        fp = int(kept.sum()) - tp
        f1 = float(_f1(tp, fp, c.n_gt - tp))
        per[cls] = {"f1": f1, "tp": tp, "fp": fp, "fn": c.n_gt - tp, "n_gt": c.n_gt}
        if c.n_gt > 0:
            num += CLASS_WEIGHTS.get(cls, 0.0) * f1
            den += CLASS_WEIGHTS.get(cls, 0.0)
    return (num / den if den else 0.0), per


def curve_where(c: ClassCurve, regions: set[str]) -> ClassCurve:
    """Та же кривая, только объекты и эталон выбранных участков. Порядок уверенности сохраняется."""
    if not c.pred_region:
        mask = np.zeros(0, dtype=bool)
    else:
        mask = np.fromiter((r in regions for r in c.pred_region), dtype=bool, count=len(c.pred_region))
    gt = Counter({r: n for r, n in c.gt_regions.items() if r in regions})
    groups = [g for g, keep in zip(c.pred_group, mask) if keep] if c.pred_group else []
    return ClassCurve(c.conf[mask], c.hit[mask], [r for r, keep in zip(c.pred_region, mask) if keep],
                      int(sum(gt.values())), gt, groups)


def score_group_thresholds(curves: dict[str, ClassCurve], taus: dict[str, dict[str, float]]) -> tuple[float, dict]:
    """Балл по кривым при пороге (группа, класс). Класс без эталона из среднего выпадает."""
    num = den = 0.0
    per = {}
    for cls, c in curves.items():
        if c.conf.size == 0:
            kept = np.zeros(0, dtype=bool)
        else:
            tau = np.fromiter((taus.get(g, {}).get(cls, 0.0) for g in c.pred_group),
                              dtype=np.float64, count=len(c.pred_group))
            kept = c.conf >= tau
        tp = int(c.hit[kept].sum()) if kept.size else 0
        fp = int(kept.sum()) - tp
        f1 = float(_f1(tp, fp, c.n_gt - tp))
        per[cls] = {"f1": f1, "tp": tp, "fp": fp, "fn": c.n_gt - tp, "n_gt": c.n_gt}
        if c.n_gt > 0:
            num += CLASS_WEIGHTS.get(cls, 0.0) * f1
            den += CLASS_WEIGHTS.get(cls, 0.0)
    return (num / den if den else 0.0), per


def bootstrap(curves: dict[str, ClassCurve], n: int = 200, seed: int = 0) -> dict[str, dict]:
    """Устойчивость порогов: регионы выбираются с возвращением, оптимум ищется заново.

    Регионы в пространстве не пересекаются, поэтому попадания из общего прохода
    переиспользуются с весами (сколько раз регион попал в выборку).
    """
    regions = sorted({r for c in curves.values() for r in c.pred_region} |
                     {r for c in curves.values() for r in c.gt_regions})
    if not regions or n <= 0:
        return {}
    idx = {r: i for i, r in enumerate(regions)}
    rng = np.random.default_rng(seed)
    taus: dict[str, list[float]] = {c: [] for c in curves}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(regions), len(regions)), minlength=len(regions)).astype(float)
        for cls, c in curves.items():
            n_gt = sum(w[idx[r]] * k for r, k in c.gt_regions.items())
            if n_gt == 0 or c.conf.size == 0:
                continue
            pw = np.array([w[idx[r]] for r in c.pred_region])
            last = np.r_[np.flatnonzero(c.conf[1:] != c.conf[:-1]), c.conf.size - 1]
            tp, nn = np.cumsum(c.hit * pw)[last], np.cumsum(pw)[last]
            taus[cls].append(float(c.conf[last[int(np.argmax(_f1(tp, nn - tp, n_gt - tp)))]]))
    return {cls: {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
                  "p90": float(np.percentile(v, 90)), "samples": len(v)}
            for cls, v in taus.items() if v}


def paired_bootstrap(a: dict[str, ClassCurve], b: dict[str, ClassCurve], tau_a: dict[str, float],
                     tau_b: dict[str, float], n: int = 1000, seed: int = 0) -> dict:
    """Разница баллов A − B на одних и тех же выборках регионов с возвращением.

    Пороги фиксированы (как в сабмите). Одни веса регионов на обе модели убирают
    из разницы разброс «какие участки попались» — сравнение парное.
    """
    regions = sorted({r for cs in (a, b) for c in cs.values() for r in c.pred_region} |
                     {r for cs in (a, b) for c in cs.values() for r in c.gt_regions})
    if not regions or n <= 0:
        return {}
    idx = {r: i for i, r in enumerate(regions)}
    rng = np.random.default_rng(seed)

    def prep(curves, taus):
        out = {}
        for cls, c in curves.items():
            kept = c.conf >= (taus.get(cls) or 0.0)
            out[cls] = (np.array([idx[r] for r in c.pred_region], np.int64)[kept], c.hit[kept],
                        {idx[r]: k for r, k in c.gt_regions.items()})
        return out

    pa, pb = prep(a, tau_a), prep(b, tau_b)
    classes = sorted(set(pa) | set(pb))
    diffs, per = [], {c: [] for c in classes}
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(regions), len(regions)), minlength=len(regions)).astype(float)
        s = []
        for prepared in (pa, pb):
            num = den = 0.0
            f1s = {}
            for cls in classes:
                if cls not in prepared:
                    continue
                reg, hit, gtr = prepared[cls]
                n_gt = sum(w[r] * k for r, k in gtr.items())
                if n_gt == 0:
                    continue
                pw = w[reg]
                tp = float((pw * hit).sum())
                f1 = float(_f1(tp, float(pw.sum()) - tp, n_gt - tp))
                f1s[cls] = f1
                num += CLASS_WEIGHTS.get(cls, 0.0) * f1
                den += CLASS_WEIGHTS.get(cls, 0.0)
            s.append((num / den if den else 0.0, f1s))
        diffs.append(s[0][0] - s[1][0])
        for cls in classes:
            if cls in s[0][1] and cls in s[1][1]:
                per[cls].append(s[0][1][cls] - s[1][1][cls])

    def summary(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "p_gt0": float((v > 0).mean())}

    return {"score": summary(diffs), "per_class": {c: summary(v) for c, v in per.items() if v}, "samples": n}


# ─────────────────────────── честный балл: порог с других семей ─────────────────

def family_folds(regions, family_of, k: int | None = None) -> list[set[str]]:
    """Фолды участков по семьям. k=None или k ≥ числа семей — одна семья на фолд.

    Семьи раскладываются жадно по размеру в k корзин, порядок детерминирован.
    Одна семья никогда не делится между фолдами.
    """
    fams: dict[str, set[str]] = {}
    for r in regions:
        fams.setdefault(family_of(r), set()).add(r)
    names = sorted(fams)
    if not k or k >= len(names):
        return [fams[f] for f in names]
    bins: list[set[str]] = [set() for _ in range(k)]
    sizes = [0] * k
    for f in sorted(names, key=lambda f: (-len(fams[f]), f)):
        i = int(np.argmin(sizes))
        bins[i] |= fams[f]
        sizes[i] += len(fams[f])
    return [b for b in bins if b]


def honest_counts(curves: dict[str, ClassCurve], folds: list[set[str]], pick: str = "max",
                  unseen: str = "drop") -> tuple[dict[str, dict[str, list[int]]], dict]:
    """TP, FP и эталоны каждого участка при порогах, подобранных без его фолда.

    Для фолда пороги (группа, класс) ищет sweep_groups на остальных участках и
    применяет к участкам фолда. Так устроен «честный балл» 0,329489: порог семьи
    не видит её эталона. unseen="drop" выбрасывает объекты группы, которой на
    остальных участках не было (порог не из чего подобрать); "keep" их оставляет.

    Возвращает ({класс: {участок: [tp, fp, n_gt]}}, сводку по фолдам).
    """
    every = set().union(*folds) if folds else set()
    counts: dict[str, dict[str, list[int]]] = {cls: {} for cls in curves}
    info = {"folds": len(folds), "unseen_dropped": 0, "unseen_kept": 0}
    for held in folds:
        train = every - held
        res = sweep_groups({cls: curve_where(c, train) for cls, c in curves.items()}, pick=pick)
        taus: dict[str, dict[str, float]] = {}
        for cls, v in res.items():
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
        for cls, c in curves.items():
            hc = curve_where(c, held)
            per = counts[cls]
            for r, n in hc.gt_regions.items():
                per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0:
                continue
            raw = [taus.get(g, {}).get(cls) for g in hc.pred_group]
            missing = np.fromiter((t is None for t in raw), dtype=bool, count=len(raw))
            fill = np.inf if unseen == "drop" else -np.inf
            tau = np.fromiter((fill if t is None else t for t in raw), dtype=np.float64, count=len(raw))
            info["unseen_dropped" if unseen == "drop" else "unseen_kept"] += int(missing.sum())
            kept = hc.conf >= tau
            for r, hit in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if hit else 1] += 1
    return counts, info


def score_counts(counts: dict[str, dict[str, list[int]]], regions: set[str] | None = None,
                 weights: dict[str, float] | None = None) -> tuple[float, dict]:
    """Итоговый балл по счётчикам участков (как в метрике: класс без эталона выпадает).

    weights — кратность участка в выборке бутстрепа; без неё каждый участок один раз.
    """
    num = den = 0.0
    per = {}
    for cls, by_region in counts.items():
        tp = fp = n_gt = 0.0
        for r, (a, b, g) in by_region.items():
            if regions is not None and r not in regions:
                continue
            w = 1.0 if weights is None else weights.get(r, 0.0)
            tp, fp, n_gt = tp + w * a, fp + w * b, n_gt + w * g
        f1 = float(_f1(tp, fp, n_gt - tp))
        per[cls] = {"f1": f1, "tp": tp, "fp": fp, "fn": n_gt - tp, "n_gt": n_gt}
        if n_gt > 0:
            num += CLASS_WEIGHTS.get(cls, 0.0) * f1
            den += CLASS_WEIGHTS.get(cls, 0.0)
    return (num / den if den else 0.0), per


def paired_bootstrap_counts(a: dict, b: dict, family_of, n: int = 1000, seed: int = 0,
                            regions: set[str] | None = None) -> dict:
    """A − B по честным счётчикам. Выборка с возвращением — семьи, не участки.

    Участки одной семьи соседствуют и делят разметчика: если тянуть участки
    по одному, разброс занижается и доля A > B выглядит увереннее, чем есть.
    """
    regs = {r for cs in (a, b) for by in cs.values() for r in by}
    if regions is not None:
        regs &= regions
    fams: dict[str, list[str]] = {}
    for r in sorted(regs):
        fams.setdefault(family_of(r), []).append(r)
    names = sorted(fams)
    if not names or n <= 0:
        return {}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        w = {r: float(k) for f, k in zip(names, pick) if k for r in fams[f]}
        diffs.append(score_counts(a, None, w)[0] - score_counts(b, None, w)[0])
    v = np.asarray(diffs)
    return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)), "p50": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "p_gt0": float((v > 0).mean()), "samples": n,
            "families": len(names)}


def apply_thresholds(fc: dict, thresholds: dict[str, float], metric_path: str | None = None) -> dict:
    """Оставляет предсказания с confidence ≥ порога своего класса."""
    norm = load_official(metric_path).normalize_class_name
    feats = [f for f in fc.get("features", [])
             if float((f.get("properties") or {}).get("confidence", 1.0))
             >= (thresholds.get(norm((f.get("properties") or {}).get("class_name"))) or 0.0)]
    return {**{k: v for k, v in fc.items() if k != "features"}, "features": feats}


def verify(pred: dict, gt: dict, thresholds: dict[str, float], metric_path: str | None = None,
           curves: dict[str, ClassCurve] | None = None) -> dict:
    """Сверка: быстрый балл против полного официального расчёта на отфильтрованных предсказаниях."""
    curves = curves or build_curves(pred, gt, metric_path)
    fast, _ = score_at(curves, thresholds)
    official = score(apply_thresholds(pred, thresholds, metric_path), gt, metric_path)["final_score"]
    return {"fast": fast, "official": official, "abs_diff": abs(fast - official)}


def sweep(pred: dict, gt: dict, metric_path: str | None = None, n_boot: int = 0) -> dict:
    """Лучший порог и F1 по каждому классу + итоговый балл с этими порогами.

    Порог ставится посередине между последним оставленным и первым отброшенным
    значением confidence — так он устойчивее переносится на новые данные.
    Классы без эталона пропускаются: по этим данным их порог не определить.
    """
    curves = build_curves(pred, gt, metric_path)
    best: dict[str, dict] = {}
    for cls in CLASSES:
        c = curves.get(cls)
        if c is None or c.n_gt == 0:
            continue
        taus, f1 = curve_table(c)
        k = int(np.argmax(f1))
        if np.isinf(taus[k]):
            tau = float(c.conf.max() + 1e-6) if c.conf.size else 1.0
        else:
            lower = c.conf[c.conf < taus[k]]
            tau = float((taus[k] + lower.max()) / 2) if lower.size else float(taus[k])
        best[cls] = {"threshold": round(tau, 6), "f1": round(float(f1[k]), 4),
                     "f1_all": round(float(f1[-1]), 4), "n_gt": c.n_gt, "n_pred": int(c.conf.size),
                     "curve": [(float(t), round(float(v), 5)) for t, v in zip(taus[1:], f1[1:])]}
    thresholds = {c: v["threshold"] for c, v in best.items()}
    est, _ = score_at(curves, thresholds)
    return {"thresholds": thresholds, "per_class": best, "estimated_score": round(est, 6),
            "bootstrap": bootstrap(curves, n_boot) if n_boot else {}, "_curves": curves}


def _filter(fc: dict, markup: set[str] | None, regions: set[str] | None) -> dict:
    feats = [f for f in fc["features"]
             if (markup is None or (f.get("properties") or {}).get("markup_type") in markup)
             and (regions is None or (f.get("properties") or {}).get("region_name") in regions)]
    return {**fc, "features": feats}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Официальная метрика и точный подбор порогов")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("--pred", required=True)
        p.add_argument("--gt", required=True)
        p.add_argument("--metric", default=None)
        p.add_argument("--markup", default=None, help="только эти markup_type, через запятую (например, li)")
        p.add_argument("--regions", default=None, help="файл со списком регионов (по одному в строке)")
        if name == "sweep":
            p.add_argument("--out", default=None, help="куда сохранить пороги (JSON)")
            p.add_argument("--bootstrap", type=int, default=0, help="повторов бутстрепа по регионам")
    p = sub.add_parser("sweep-groups", help="совместные пороги по (группа порога, класс) на общий F1 класса")
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--regions", default=None)
    p.add_argument("--out", default=None, help="JSON {группа: {класс: порог}} для [thresholds_by_markup]")
    p = sub.add_parser("compare", help="A против B: пороги — оптимум каждой, разница — парный бутстреп по регионам")
    p.add_argument("--pred", required=True, help="A")
    p.add_argument("--pred-b", required=True, help="B")
    p.add_argument("--gt", required=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--markup", default=None)
    p.add_argument("--regions", default=None)
    p.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8-sig"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8-sig"))
    markup = set(args.markup.split(",")) if getattr(args, "markup", None) else None
    regions = ({ln.strip() for ln in Path(args.regions).read_text(encoding="utf-8").splitlines() if ln.strip()}
               if args.regions else None)
    pred, gt = _filter(pred, markup, regions), _filter(gt, markup, regions)
    if args.cmd == "sweep-groups":
        curves = build_curves(pred, gt, args.metric)
        res = sweep_groups(curves)
        taus: dict[str, dict[str, float]] = {}
        for cls, v in res.items():
            print(f"{cls:24s} GT {v['n_gt']:5d}  F1 {v['f1']:.3f}")
            for g, gv in v["groups"].items():
                taus.setdefault(g, {})[cls] = gv["threshold"]
                print(f"    {g:8s} порог {gv['threshold']:.6f}  оставлено {gv['kept']:6d}  TP {gv['tp']:5d}")
        est = official_weighted(res)
        official = score(apply_group_thresholds(pred, taus, args.metric), gt, args.metric)["final_score"]
        print(f"балл по кривым {est:.6f}; официальный расчёт {official:.6f}")
        if args.out:
            Path(args.out).write_text(json.dumps(taus, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if args.cmd == "compare":
        pred_b = _filter(json.loads(Path(args.pred_b).read_text(encoding="utf-8-sig")), markup, regions)
        ra, rb = sweep(pred, gt, args.metric), sweep(pred_b, gt, args.metric)
        res = paired_bootstrap(ra["_curves"], rb["_curves"], ra["thresholds"], rb["thresholds"], args.boot)
        print(f"балл A {ra['estimated_score']:.4f}, B {rb['estimated_score']:.4f}")
        s = res["score"]
        print(f"A − B: среднее {s['mean']:+.4f}, 90% интервал [{s['p5']:+.4f}; {s['p95']:+.4f}], "
              f"доля выборок с A > B {s['p_gt0']:.2f}")
        for cls, v in res["per_class"].items():
            fa, fb = ra["per_class"][cls]["f1"], rb["per_class"][cls]["f1"]
            print(f"  {cls:24s} F1 {fa:.3f} − {fb:.3f}: [{v['p5']:+.3f}; {v['p95']:+.3f}], A > B в {v['p_gt0']:.2f}")
        return
    if args.cmd == "score":
        res = score(pred, gt, args.metric)
        print(json.dumps({"final_score": res["final_score"],
                          "per_class": {c: {k: v[k] for k in ("f1", "tp", "fp", "fn")}
                                        for c, v in res["per_class"].items()}},
                         ensure_ascii=False, indent=2))
        return
    res = sweep(pred, gt, args.metric, args.bootstrap)
    check = verify(pred, gt, res["thresholds"], args.metric, res["_curves"])
    all_in = score_at(res["_curves"], {c: 0.0 for c in res["_curves"]})[0]
    print(f"{'класс':24s} {'GT':>6s} {'pred':>6s} {'порог':>8s} {'F1':>6s} {'F1 всё':>7s}  бутстреп p10–p50–p90")
    for cls, v in res["per_class"].items():
        b = res["bootstrap"].get(cls)
        bt = f"{b['p10']:.3f}–{b['p50']:.3f}–{b['p90']:.3f}" if b else "—"
        print(f"{cls:24s} {v['n_gt']:6d} {v['n_pred']:6d} {v['threshold']:8.4f} {v['f1']:6.3f} {v['f1_all']:7.3f}  {bt}")
    print(f"балл при порогах {res['estimated_score']:.6f}; официальный расчёт {check['official']:.6f} "
          f"(расхождение {check['abs_diff']:.1e}); без порогов {all_in:.6f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res["thresholds"], indent=2), encoding="utf-8")
        print(f"пороги → {args.out} (перенесите в configs/default.toml, секция [thresholds])")


if __name__ == "__main__":
    _cli()
