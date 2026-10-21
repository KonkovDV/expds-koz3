"""Пометка лидара с грубой сеткой DEM.

Предсказания вне обучения не хранят имя LAS. Облако определяется по попаданию
точки в рамку заголовка; шаг сетки считается так же, как на инференсе.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .accel import grid_spec
from .catalog import scan_root
from .crs import WEB_MERCATOR, transformer


def first_xy(geom: dict) -> tuple[float, float]:
    c = geom["coordinates"]
    while c and isinstance(c[0], list):
        c = c[0]
    return float(c[0]), float(c[1])


def las_boxes(root: str | Path, res: float = 0.1, max_pixels: int = 60_000_000) -> dict[str, list[tuple]]:
    """Регион → рамки LAS (minx, miny, maxx, maxy, factor, имя файла)."""
    boxes: dict[str, list[tuple]] = {}
    for reg in scan_root(root):
        if not reg.las:
            continue
        rows = []
        import laspy
        for las in reg.las:
            with laspy.open(las) as src:
                h = src.header
            spec = grid_spec(float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]),
                             res, max_pixels)
            rows.append((float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]),
                         int(spec["factor"]), las.name))
        boxes[reg.name] = rows
    return boxes


def tag(fc: dict, boxes: dict[str, list[tuple]]) -> int:
    """Ставит threshold_group=li_coarse объектам li внутри грубого облака. Уже помеченные не трогает."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, feat in enumerate(fc["features"]):
        props = feat["properties"]
        if props.get("markup_type") != "li" or props.get("threshold_group"):
            continue
        groups[(props.get("region_name") or "", props.get("original_crs") or "")].append(i)
    n = 0
    for (region, crs), idxs in groups.items():
        rows = boxes.get(region) or []
        if not rows:
            continue
        if len(rows) == 1:
            factors = np.full(len(idxs), rows[0][4])
        else:
            xs, ys = zip(*(first_xy(fc["features"][i]["geometry"]) for i in idxs))
            if crs and crs != WEB_MERCATOR:
                xs, ys = transformer(WEB_MERCATOR, crs).transform(np.array(xs), np.array(ys))
                xs, ys = np.asarray(xs), np.asarray(ys)
            else:
                xs, ys = np.asarray(xs), np.asarray(ys)
            factors = np.ones(len(idxs), np.int32)
            claimed = np.zeros(len(idxs), bool)
            for minx, miny, maxx, maxy, factor, _name in rows:
                inside = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy) & ~claimed
                factors[inside] = factor
                claimed |= inside
        for i, factor in zip(idxs, factors):
            if int(factor) > 1:
                fc["features"][i]["properties"]["threshold_group"] = "li_coarse"
                n += 1
    return n
