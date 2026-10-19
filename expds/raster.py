"""Работа с растрами: чтение с нужным шагом, окна, полигонизация, круги.

Ортофото и аэрофото бывают в сотни мегапикселей, поэтому всё читается
окнами (rasterio.windows) и сразу приводится к рабочему шагу (например, 0,5 м).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import Point, shape

from .crs import WEB_MERCATOR, to_crs


@dataclass
class Tile:
    """Окно растра, приведённое к рабочему шагу."""
    data: np.ndarray          # C×H×W
    transform: object         # аффинное преобразование тайла (рабочий шаг)
    crs: str | None
    window: Window            # окно в пикселях исходного растра


def raster_res(path: str | Path) -> float:
    """Размер пикселя исходного растра (в единицах CRS, обычно метры)."""
    with rasterio.open(path) as ds:
        return float(abs(ds.res[0]))


def iter_tiles(path: str | Path, target_res: float, tile: int = 256, overlap: float = 0.25,
               bands: list[int] | None = None) -> Iterator[Tile]:
    """Скользящее окно по растру: тайлы tile×tile пикселей рабочего шага с перекрытием."""
    with rasterio.open(path) as ds:
        scale = target_res / abs(ds.res[0])          # исходных пикселей на рабочий
        src_tile = max(1, int(round(tile * scale)))
        step = max(1, int(round(src_tile * (1 - overlap))))
        idx = bands or list(range(1, min(ds.count, 3) + 1))
        crs = ds.crs.to_string() if ds.crs else None
        rows = list(range(0, max(ds.height - src_tile, 0) + 1, step)) or [0]
        cols = list(range(0, max(ds.width - src_tile, 0) + 1, step)) or [0]
        if rows[-1] + src_tile < ds.height:
            rows.append(max(ds.height - src_tile, 0))
        if cols[-1] + src_tile < ds.width:
            cols.append(max(ds.width - src_tile, 0))
        for r in rows:
            for c in cols:
                win = Window(c, r, min(src_tile, ds.width - c), min(src_tile, ds.height - r))
                out_h = max(1, int(round(win.height / scale)))
                out_w = max(1, int(round(win.width / scale)))
                data = ds.read(idx, window=win, out_shape=(len(idx), out_h, out_w),
                               resampling=Resampling.bilinear if scale < 1 else Resampling.average)
                tr = ds.window_transform(win) * rasterio.Affine.scale(win.width / out_w, win.height / out_h)
                yield Tile(data=data, transform=tr, crs=crs, window=win)


def mask_to_polygons(labels: np.ndarray, transform, crs: str | None,
                     simplify: float = 0.0) -> list[tuple[object, int]]:
    """Растровые экземпляры (0 — фон) → [(полигон в EPSG:3857, номер экземпляра)]."""
    out = []
    labels = labels.astype(np.int32)
    for geom, val in shapes(labels, mask=labels > 0, transform=transform, connectivity=8):
        poly = shape(geom)
        if simplify > 0:
            poly = poly.simplify(simplify, preserve_topology=True)
        if crs:
            poly = to_crs(poly, crs, WEB_MERCATOR)
        out.append((poly, int(val)))
    return out


def circle_3857(x: float, y: float, radius_m: float, crs: str | None, n: int = 16):
    """Круг радиуса radius_m вокруг точки (x, y) в CRS растра → полигон EPSG:3857.

    Метрика засчитывает точечный эталон по расстоянию от центроида до точки
    (≤ 2 м), а полигональный — по покрытию и внутренней точке, поэтому
    маленький круг вокруг центра объекта — допустимый и честный выход.
    """
    poly = Point(x, y).buffer(radius_m, quad_segs=max(2, n // 4))
    return to_crs(poly, crs, WEB_MERCATOR) if crs else poly


def pixel_to_xy(transform, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центры пикселей → координаты CRS растра."""
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return np.asarray(xs), np.asarray(ys)


def meters_to_px(meters: float, res: float) -> int:
    """Длина в метрах → целое число пикселей (не меньше 1)."""
    return max(1, int(math.ceil(meters / res)))
