"""Кандидаты → объекты GeoJSON в соглашениях bot-2/repo/expds/writer.py (EPSG:3857, только Polygon,
class_name, confidence ∈ [0,1] с 5 знаками, region_name, crs, markup_type, original_crs; координаты 0,01).

Круг строится в РОДНОЙ метрической СК (истинный радиус в метрах), затем вершины → EPSG:3857.
"""
from __future__ import annotations

import math

import numpy as np

from .config import DEFAULTS, HM_CLASSES

MAX_VERTICES_HINT = 33   # 32 вершины + замыкание; фактический лимит — expds.constants.MAX_VERTICES


def candidates_to_feats(t: dict, region_name: str, native_crs: str, score_key: str = "score",
                        class_names=HM_CLASSES, cfg: dict = DEFAULTS, min_conf: float = 0.0) -> list[dict]:
    """→ list[dict] {"geom": shapely Polygon в EPSG:3857, "class_name", "confidence", "region_name",
    "markup_type", "original_crs"} — вход для expds.writer.write_geojson/to_feature_collection или to_fc ниже."""
    from pyproj import Transformer
    from shapely.geometry import Polygon
    X, Y, c = np.asarray(t["X"]), np.asarray(t["Y"]), np.asarray(t["cls"])
    s = np.asarray(t[score_key], np.float64)
    n, r = len(X), float(cfg["circle_r_m"])
    k = int(cfg["circle_points"])
    if n == 0:
        return []
    a = 2 * math.pi * np.arange(k) / k
    px = (X[:, None] + r * np.cos(a)[None]).ravel()
    py = (Y[:, None] + r * np.sin(a)[None]).ravel()
    tr = Transformer.from_crs(native_crs, "EPSG:3857", always_xy=True)
    mx, my = tr.transform(px, py)
    mx, my = np.asarray(mx).reshape(n, k), np.asarray(my).reshape(n, k)
    out = []
    for i in np.argsort(-s, kind="stable"):
        if s[i] < min_conf:
            continue
        out.append({"geom": Polygon(np.column_stack([mx[i], my[i]])), "class_name": class_names[int(c[i])],
                    "confidence": float(min(1.0, max(0.0, s[i]))), "region_name": region_name,
                    "markup_type": cfg["markup_type"], "original_crs": native_crs})
    return out


def to_fc(feats: list[dict]) -> dict:
    """Автономная копия expds.writer.to_feature_collection (те же поля и округления), чтобы не тянуть expds."""
    from shapely.geometry import mapping
    out = []
    for k, f in enumerate(feats):
        geom = mapping(f["geom"])
        geom["coordinates"] = [[[round(x, 2), round(y, 2)] for x, y, *_ in ring] for ring in geom["coordinates"]]
        props = {"fid": k, "class_name": f["class_name"],
                 "confidence": round(min(1.0, max(0.0, float(f.get("confidence", 1.0)))), 5),
                 "region_name": f.get("region_name", "default_source"), "crs": "EPSG:3857"}
        if f.get("markup_type"):
            props["markup_type"] = f["markup_type"]
        if f.get("original_crs"):
            props["original_crs"] = f["original_crs"]
        out.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": out}
