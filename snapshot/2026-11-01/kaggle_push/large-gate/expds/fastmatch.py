"""Быстрый путь сопоставления кандидатов с эталоном для гейтов (EXPDS_RUST_MATCH=1).

Rust (expds_core.match_prefilter: R-дерево rstar + площадь пересечения geo, rayon) отбрасывает пары,
которые заведомо не дают hit официального `_match_quality` (cov < thr − EPS_COV или
dist(centroid, pt) > r + EPS_D). На оставшихся парах hit/score считает сам официальный
`calc._match_quality`, поэтому списки hit и оценки побайтно совпадают с полным перебором STRtree
(tests/test_w0_fastmatch.py; ошибка geo ~1e-12 ≪ EPS_COV). Без модуля/флага — None, вызывающий
использует прежний путь shapely.STRtree.
"""
from __future__ import annotations

import os

import numpy as np

EPS_COV = float(os.environ.get("EXPDS_RUST_MATCH_EPS", "1e-4"))
EPS_D = 1e-3  # м
MIN_PAIRS = int(os.environ.get("EXPDS_RUST_MATCH_MIN_PAIRS", "2000"))


def enabled() -> bool:
    if os.environ.get("EXPDS_RUST_MATCH", "0") != "1":
        return False
    from . import accel
    return accel._rs is not None and hasattr(accel._rs, "match_prefilter")


def _ragged(geoms):
    import shapely
    if not len(geoms):
        return np.zeros((0, 2)), np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros(1, np.int64)
    t, xy, off = shapely.to_ragged_array(np.asarray(geoms, dtype=object))
    off = [np.ascontiguousarray(o, np.int64) for o in off]
    if t == shapely.GeometryType.POLYGON:
        ring, geom = off
        part, geom = geom, np.arange(len(geoms) + 1, dtype=np.int64)
    elif t == shapely.GeometryType.MULTIPOLYGON:
        ring, part, geom = off
    else:
        raise TypeError(t)
    return np.ascontiguousarray(xy[:, :2], np.float64), ring, part, geom


def prefilter(pred_geoms, gt_geoms, calc):
    """Замена `STRtree(gt').query(pred_geoms)` (gt' — точки с буфером r): (q_pred, q_gt), отсортированы
    по (pred, gt); надмножество всех пар с hit=True. None — быстрый путь недоступен."""
    if not enabled():
        return None
    import shapely
    from . import accel
    try:
        pa = np.asarray(pred_geoms, dtype=object)
        ga = np.asarray(gt_geoms, dtype=object)
        pt = shapely.get_type_id(pa)
        gt_t = shapely.get_type_id(ga)
        pidx = np.flatnonzero((pt == 3) & ~shapely.is_empty(pa))   # остальные официальный цикл пропускает
        gpoly = np.flatnonzero(((gt_t == 3) | (gt_t == 6)) & ~shapely.is_empty(ga))
        gpts = np.flatnonzero((gt_t == 0) & ~shapely.is_empty(ga))
        # грубый отбор по рамкам тем же STRtree (C), в Rust уходят только задетые кандидаты
        tree = shapely.STRtree([g.buffer(calc.r_tolerance) if t == 0 else g for g, t in zip(ga, gt_t)])
        q0, g0 = tree.query(pa[pidx])
        if len(q0) < MIN_PAIRS:   # мелкие наборы: накладные расходы Rust больше выигрыша
            return pidx[q0], g0
        pidx = pidx[np.unique(q0)]
        P = _ragged(pa[pidx])
        G = _ragged(ga[gpoly])
        ptxy = np.ascontiguousarray(shapely.get_coordinates(ga[gpts])[:, :2], np.float64).reshape(-1, 2)
        qp, qg = accel._rs.match_prefilter(*P, *G, gpoly.astype(np.int64), ptxy, gpts.astype(np.int64),
                                           float(calc.r_tolerance), float(calc.coverage_threshold), EPS_COV, EPS_D)
        return pidx[qp], qg
    except Exception:
        return None
