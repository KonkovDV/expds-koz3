"""Классические детекторы геофизики (MagGr, EtGr, TerFl) — без видеокарты.

finds_points: точечные аномалии магниторазведки — экстремумы высокочастотной
составляющей |z − фон| выше k·MAD. Выход — маленький круг (центроид ≤ 2 м от
эталонной точки засчитывается).

object_poly: площадные аномалии — связные области, где сглаженная
нормированная аномалия выше порога, с фильтром по площади.

Обучающих участков геофизики мало (три «семьи»), поэтому начинать стоит с
классики; нейросеть — только если выиграет на проверке по «семьям».
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import accel
from .raster import circle_3857, mask_to_polygons, pixel_to_xy

log = logging.getLogger(__name__)


@dataclass
class GeoParams:
    """Параметры классики; подбираются на трейне официальной метрикой."""
    background_m: float = 4.0       # радиус фона для высоких частот
    peak_radius_m: float = 1.0      # подавление соседних пиков
    peak_k: float = 4.0             # порог пика, в MAD
    point_circle_m: float = 0.3
    poly_k: float = 2.5             # порог области, в MAD
    smooth_m: float = 0.5
    min_area_m2: float = 0.3
    max_area_m2: float = 60.0


def read_band(path: str | Path) -> tuple[np.ndarray, object, str | None, float]:
    """Первый канал как float32 с NaN вместо nodata."""
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
        crs = ds.crs.to_string() if ds.crs else None
        return a, ds.transform, crs, float(abs(ds.res[0]))


def robust_z(a: np.ndarray) -> np.ndarray:
    """(a − медиана) / (1,4826·MAD) по конечным значениям."""
    fin = np.isfinite(a)
    if not fin.any():
        return np.full_like(a, np.nan)
    med = float(np.median(a[fin]))
    mad = float(np.median(np.abs(a[fin] - med))) * 1.4826 or 1.0
    return ((a - med) / mad).astype(np.float32)


def finds_points(path: str | Path, region: str, markup_type: str, p: GeoParams | None = None) -> list[dict]:
    """Точечные аномалии → круги класса finds_points."""
    p = p or GeoParams()
    a, tr, crs, res = read_band(path)
    z = robust_z(a)
    hp = z - accel.box_mean(z, max(1, int(round(p.background_m / res))))
    mag = np.where(np.isfinite(hp), np.abs(hp), np.nan).astype(np.float32)
    rows, cols, vals = accel.find_peaks(mag, max(1, int(round(p.peak_radius_m / res))), p.peak_k)
    xs, ys = pixel_to_xy(tr, rows, cols)
    out = []
    for x, y, v in zip(xs, ys, vals):
        conf = float(1.0 - np.exp(-(v - p.peak_k + 1.0) / 3.0))
        out.append({"geom": circle_3857(float(x), float(y), p.point_circle_m, crs), "class_name": "finds_points",
                    "confidence": max(0.01, min(conf, 0.99)), "region_name": region,
                    "markup_type": markup_type, "source": Path(path).name, "original_crs": crs})
    return out


def object_polys(path: str | Path, region: str, markup_type: str, p: GeoParams | None = None) -> list[dict]:
    """Площадные аномалии → полигоны класса object_poly."""
    p = p or GeoParams()
    from scipy.ndimage import label as cc_label

    a, tr, crs, res = read_band(path)
    z = robust_z(a)
    hp = z - accel.box_mean(z, max(1, int(round(p.background_m / res))))
    sm = accel.box_mean(np.abs(hp).astype(np.float32), max(1, int(round(p.smooth_m / res))))
    mask = np.isfinite(sm) & (sm >= p.poly_k)
    lab, n = cc_label(mask)
    if n == 0:
        return []
    px_area = res * res
    sizes = np.bincount(lab.ravel())
    keep = (sizes * px_area >= p.min_area_m2) & (sizes * px_area <= p.max_area_m2)
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    peak = np.zeros(n + 1, np.float32)
    np.maximum.at(peak, lab.ravel(), np.nan_to_num(sm, nan=0).ravel())
    out = []
    for poly, k in mask_to_polygons(lab, tr, crs):
        conf = float(1.0 - np.exp(-(peak[k] - p.poly_k + 0.5) / 2.0))
        out.append({"geom": poly, "class_name": "object_poly", "confidence": max(0.01, min(conf, 0.99)),
                    "region_name": region, "markup_type": markup_type, "source": Path(path).name,
                    "original_crs": crs})
    return out
