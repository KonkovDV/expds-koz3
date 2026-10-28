"""GT GeoJSON → objs в пикселях кропа/участка (правило центра и диаметра — как region_targets у bot-9)."""
from __future__ import annotations

import math

import numpy as np

from .config import DEFAULTS, HM_CLASSES


def transformer(src: str, dst: str):
    from pyproj import Transformer
    return Transformer.from_crs(src, dst, always_xy=True)


def reproject(geom, src: str, dst: str):
    from shapely.ops import transform
    if src == dst:
        return geom
    t = transformer(src, dst)
    return transform(t.transform, geom)


def center_and_diam(geom, default_diam_m: float):
    """Полигон: центроид (или representative_point, если центроид снаружи), D = 2√(S/π). Точка: D = default."""
    if geom.geom_type == "Point":
        return geom.x, geom.y, float(default_diam_m)
    c = geom.centroid
    if not geom.contains(c):
        c = geom.representative_point()
    return c.x, c.y, 2.0 * math.sqrt(max(geom.area, 0.0) / math.pi)


def gt_to_objs(features, left: float, top: float, step: float, src_crs: str | None = None, dst_crs: str | None = None,
               classes=HM_CLASSES, cfg: dict = DEFAULTS, markup_type: str | None = "li") -> np.ndarray:
    """features — список GeoJSON Feature (properties.class_name, markup_type). Геометрия переводится src_crs→dst_crs
    (метрическая СК растра), затем x = (X−left)/step, y = (top−Y)/step, diam_px = D/step.
    → float64 (N,4) [x, y, diam_px, cls]."""
    from shapely.geometry import shape
    t = transformer(src_crs, dst_crs) if (src_crs and dst_crs and src_crs != dst_crs) else None
    out = []
    for f in features:
        p = f.get("properties") or {}
        if p.get("class_name") not in classes:
            continue
        if markup_type and p.get("markup_type", markup_type) != markup_type:
            continue
        g = shape(f["geometry"])
        if t is not None:
            from shapely.ops import transform
            g = transform(t.transform, g)
        X, Y, D = center_and_diam(g, cfg["default_diam_m"])
        out.append(((X - left) / step, (top - Y) / step, D / step, classes.index(p["class_name"])))
    return np.asarray(out, np.float64).reshape(-1, 4)


def crop_objs(objs: np.ndarray, x0: int, y0: int, S: int, margin_px: float = 0.0) -> np.ndarray:
    """objs участка → objs кропа [x0, x0+S)×[y0, y0+S); оставляет и соседей за краем на margin+D (для хвостов)."""
    o = np.asarray(objs, np.float64).reshape(-1, 4).copy()
    o[:, 0] -= x0
    o[:, 1] -= y0
    m = margin_px + o[:, 2]
    keep = (o[:, 0] > -m) & (o[:, 0] < S + m) & (o[:, 1] > -m) & (o[:, 1] < S + m)
    return o[keep]
