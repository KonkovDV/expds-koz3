"""Сдвиг одного слоя разметки (или снимка) относительно другого.

Метрика засчитывает полигон, только если внутренняя точка предсказания лежит
внутри эталона. Курган на снимке — круг радиусом 4–12 м. Если снимок 1940-х или
спутниковая сцена привязаны со сдвигом в несколько метров, эталон этого слоя
нарисован по снимку и сдвинут относительно земли (облака LAS) на тот же вектор.
Тогда копия кургана лидара в слой снимка промахивается, а сеть, обученная на
таком снимке, учит смещённую точку.

Сдвиг оценивается по парам объектов одного класса в двух слоях: мода всех
смещений ближе r0 (устойчиво к густым полям курганов), затем уточнение медианой
по ближайшим парам ближе r1. dst ≈ ref + (dx, dy) в метрах UTM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .crs import WEB_MERCATOR, transformer, utm_for_lonlat


@dataclass
class Shift:
    """Перенос dst относительно ref, метры UTM. systematic — устойчив и больше шума."""

    dx: float = 0.0
    dy: float = 0.0
    n_pairs: int = 0
    mad_m: float = 0.0
    se_m: float = 0.0
    systematic: bool = False

    @property
    def norm(self) -> float:
        return float(np.hypot(self.dx, self.dy))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shift_m"] = round(self.norm, 3)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def utm_of(xs_3857: np.ndarray, ys_3857: np.ndarray) -> str:
    """Зона UTM по центру набора точек EPSG:3857."""
    lon, lat = transformer(WEB_MERCATOR, "EPSG:4326").transform(float(np.median(xs_3857)),
                                                                   float(np.median(ys_3857)))
    return utm_for_lonlat(lon, lat)


def _mode_offset(ref: np.ndarray, dst: np.ndarray, r0: float, bin_m: float) -> tuple[float, float] | None:
    """Самое частое смещение dst − ref среди всех пар ближе r0 (гистограмма, сглаженная 3×3)."""
    from scipy.ndimage import uniform_filter
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    offs = []
    for i, near in enumerate(tree.query_ball_point(ref, r0)):
        if near:
            offs.append(dst[near] - ref[i])
    if not offs:
        return None
    d = np.concatenate(offs)
    n = int(np.ceil(2 * r0 / bin_m))
    h, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=n, range=[[-r0, r0], [-r0, r0]])
    h = uniform_filter(h, size=3, mode="constant")
    ix, iy = np.unravel_index(int(np.argmax(h)), h.shape)
    return float((ex[ix] + ex[ix + 1]) / 2), float((ey[iy] + ey[iy + 1]) / 2)


def estimate_shift(ref: np.ndarray, dst: np.ndarray, r0: float = 30.0, r1: float = 6.0, iters: int = 4,
                   min_pairs: int = 5, min_shift_m: float = 1.5, bin_m: float = 1.0) -> Shift:
    """Перенос dst относительно ref по точкам N×2 и M×2 в метрах.

    systematic: пар не меньше min_pairs, |сдвиг| ≥ min_shift_m и больше трёх стандартных ошибок
    (1,4826·MAD/√n по каждой оси).
    """
    from scipy.spatial import cKDTree

    ref = np.asarray(ref, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(ref) == 0 or len(dst) == 0:
        return Shift()
    start = _mode_offset(ref, dst, r0, bin_m)
    if start is None:
        return Shift()
    t = np.array(start)
    tree = cKDTree(dst)
    d = np.zeros((0, 2))
    for _ in range(max(1, iters)):
        dist, idx = tree.query(ref + t, distance_upper_bound=r1)
        ok = np.isfinite(dist) & (idx < len(dst))
        if ok.sum() == 0:
            return Shift()
        d = dst[idx[ok]] - ref[ok]
        t_new = np.median(d, axis=0)
        if np.allclose(t_new, t, atol=1e-3):
            t = t_new
            break
        t = t_new
    resid = d - t
    mad = np.median(np.abs(resid), axis=0)
    se = 1.4826 * mad / np.sqrt(max(len(d), 1))
    se_norm = float(np.hypot(*se))
    norm = float(np.hypot(*t))
    systematic = len(d) >= min_pairs and norm >= min_shift_m and norm > 3 * max(se_norm, 1e-6)
    return Shift(float(t[0]), float(t[1]), len(d), float(np.hypot(*mad)), se_norm, bool(systematic))


def contain_rate(points: np.ndarray, polys: list, shift: tuple[float, float] = (0.0, 0.0)) -> float:
    """Доля точек (метры), которые после переноса лежат внутри хотя бы одного полигона (те же метры)."""
    import shapely
    from shapely.strtree import STRtree

    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) == 0 or not polys:
        return 0.0
    moved = shapely.points(pts[:, 0] + shift[0], pts[:, 1] + shift[1])
    tree = STRtree(polys)
    hit = np.zeros(len(pts), bool)
    src, _ = tree.query(moved, predicate="intersects")
    hit[np.unique(src)] = True
    return float(hit.mean())


def shift_geoms_3857(geoms: list, dx: float, dy: float, utm: str) -> list:
    """Перенос геометрий EPSG:3857 на (dx, dy) метров зоны utm."""
    from shapely import affinity

    from .crs import to_crs

    return [to_crs(affinity.translate(to_crs(g, WEB_MERCATOR, utm), dx, dy), utm, WEB_MERCATOR) for g in geoms]


def blob_response(img: np.ndarray, radius_px: float) -> np.ndarray:
    """|DoG| на масштабе насыпи радиусом radius_px (σ = r/√2, второй σ в 1,6 раза шире), в MAD.

    Пятно почвы распаханного кургана и тень насыпи дают сильный отклик в центре; ровное поле — слабый.
    NaN (нет данных) заполняется медианой и в ответе остаётся NaN.
    """
    from scipy.ndimage import gaussian_filter

    a = np.asarray(img, np.float32)
    bad = ~np.isfinite(a)
    if bad.all():
        return np.full(a.shape, np.nan, np.float32)
    a = np.where(bad, np.nanmedian(a), a)
    s = max(radius_px / np.sqrt(2.0), 0.7)
    dog = np.abs(gaussian_filter(a, s) - gaussian_filter(a, 1.6 * s))
    med = float(np.median(dog[~bad]))
    mad = float(np.median(np.abs(dog[~bad] - med))) or 1e-6
    out = ((dog - med) / (1.4826 * mad)).astype(np.float32)
    out[bad] = np.nan
    return out


def label_shift_scan(resp: np.ndarray, rows: np.ndarray, cols: np.ndarray, max_px: int,
                     step_px: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Средний отклик в точках эталона при каждом сдвиге (dx, dy) в пикселях: (сдвиги K×2, значения K)."""
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    h, w = resp.shape
    offs, vals = [], []
    for dy in range(-max_px, max_px + 1, step_px):
        for dx in range(-max_px, max_px + 1, step_px):
            r, c = rows + dy, cols + dx
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            v = resp[r[ok], c[ok]] if ok.any() else np.array([np.nan])
            offs.append((dx, dy))
            vals.append(float(np.nanmean(v)) if np.isfinite(v).any() else np.nan)
    return np.asarray(offs, np.int64), np.asarray(vals, np.float64)


def label_shift(resp: np.ndarray, rows: np.ndarray, cols: np.ndarray, max_px: int, px_m: float,
                min_shift_m: float = 1.5, min_z: float = 4.0) -> dict:
    """Сдвиг, при котором точки эталона сильнее всего садятся на пятна снимка.

    z — насколько пик выше медианы по всем сдвигам (в MAD этих значений). systematic: пик не в нуле
    (≥ min_shift_m), z ≥ min_z и пик выше отклика без сдвига на два MAD.
    dx — на восток по столбцам, dy — на юг по строкам растра, метры.
    """
    offs, vals = label_shift_scan(resp, rows, cols, max_px)
    ok = np.isfinite(vals)
    if not ok.any():
        return {"n": len(rows), "systematic": False}
    v = vals[ok]
    o = offs[ok]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) or 1e-6
    k = int(np.argmax(v))
    zero = vals[(offs[:, 0] == 0) & (offs[:, 1] == 0)]
    v0 = float(zero[0]) if zero.size and np.isfinite(zero[0]) else med
    dx, dy = float(o[k, 0] * px_m), float(o[k, 1] * px_m)
    z_peak = (float(v[k]) - med) / (1.4826 * mad)
    z_zero = (v0 - med) / (1.4826 * mad)
    systematic = bool(np.hypot(dx, dy) >= min_shift_m and z_peak >= min_z and v[k] > v0 + 2 * 1.4826 * mad)
    return {"n": len(rows), "dx_m": round(dx, 2), "dy_m": round(dy, 2),
            "shift_m": round(float(np.hypot(dx, dy)), 2), "peak": round(float(v[k]), 3),
            "at_zero": round(v0, 3), "z_peak": round(z_peak, 2), "z_zero": round(z_zero, 2),
            "systematic": systematic}


def raster_label_shift(path, geoms: list, res_m: float = 1.0, max_shift_m: float = 20.0,
                       fallback_crs: str | None = None, radius_m: float | None = None,
                       min_points: int = 8) -> dict:
    """Сдвиг точек geoms (EPSG:3857) относительно пятен снимка path.

    Снимок читается на шаге res_m метров на земле, каналы усредняются; масштаб пятна — radius_m
    или радиус медианного полигона geoms. dx_m — на восток, dy_m — на юг, dn_m = −dy_m — на север.
    Пустой словарь — снимок не читается; меньше min_points точек на снимке — systematic False.
    """
    import math

    import rasterio
    from rasterio.enums import Resampling

    from .crs import to_crs, usable_raster_crs
    from .raster import open_metric, raster_kind_and_latitude

    if not geoms:
        return {}
    pts = [g.representative_point() for g in geoms]
    if radius_m is None:
        utm = utm_of(np.array([p.x for p in pts]), np.array([p.y for p in pts]))
        radius_m = math.sqrt(float(np.median([to_crs(g, WEB_MERCATOR, utm).area for g in geoms])) / math.pi)
    with open_metric(path) as ds:
        declared = ds.crs.to_string() if ds.crs else None
        crs = usable_raster_crs(declared, ds.bounds, fallback_crs)
        if crs is None:
            return {}
        kind, lat = raster_kind_and_latitude(rasterio.crs.CRS.from_user_input(crs), ds.transform,
                                             ds.width, ds.height)
        scale = math.cos(math.radians(lat)) if kind == "3857" else 1.0
        native_m = abs(ds.transform.a) * scale
        f = max(1, round(res_m / native_m))
        h, w = max(1, ds.height // f), max(1, ds.width // f)
        n = min(ds.count, 3)
        a = ds.read(list(range(1, n + 1)), out_shape=(n, h, w), resampling=Resampling.average).astype(np.float32)
        tr = ds.transform @ ds.transform.scale(ds.width / w, ds.height / h)
    valid = (a != 0).any(axis=0)
    gray = np.where(valid, a.mean(axis=0), np.nan)
    px_m = abs(tr.a) * scale
    xs, ys = transformer(WEB_MERCATOR, crs).transform(np.array([p.x for p in pts]), np.array([p.y for p in pts]))
    cols, rows = ~tr @ (np.asarray(xs), np.asarray(ys))
    rows, cols = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    ok[ok] &= valid[rows[ok], cols[ok]]
    if ok.sum() < min_points:
        return {"n": int(ok.sum()), "systematic": False, "note": f"меньше {min_points} точек на снимке"}
    resp = blob_response(gray, radius_m / px_m)
    row = label_shift(resp, rows[ok], cols[ok], max(1, math.ceil(max_shift_m / px_m)), px_m)
    if "dy_m" in row:
        row["dn_m"] = round(-row["dy_m"], 2)
    return {**row, "px_m": round(px_m, 3), "kurgan_radius_m": round(radius_m, 2)}
