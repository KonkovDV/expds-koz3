"""Работа с растрами: чтение с нужным шагом, окна, полигонизация, круги.

Ортофото и аэрофото бывают в сотни мегапикселей, поэтому всё читается
окнами (rasterio.windows) и сразу приводится к рабочему шагу (например, 0,5 м).
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import Point

from .crs import WEB_MERCATOR, to_crs

log = logging.getLogger(__name__)


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
    """Растровые экземпляры (0 — фон) → [(полигон в EPSG:3857, номер экземпляра)].

    Кольца GDAL собираются в один массив, и полигоны, упрощение и пересчёт CRS
    делаются одним вызовом shapely на все объекты. Геометрия та же, что у
    shape() + simplify() + to_crs() по одному полигону.
    """
    import shapely

    labels = labels.astype(np.int32)
    coords: list[list] = []
    ring_ends: list[int] = []
    poly_ends: list[int] = []
    vals: list[int] = []
    n_pts = 0
    for geom, val in shapes(labels, mask=labels > 0, transform=transform, connectivity=8):
        for ring in geom["coordinates"]:
            coords.append(ring)
            n_pts += len(ring)
            ring_ends.append(n_pts)
        poly_ends.append(len(ring_ends))
        vals.append(int(val))
    if not vals:
        return []
    xy = np.concatenate([np.asarray(ring, dtype=np.float64) for ring in coords])
    polys = shapely.from_ragged_array(shapely.GeometryType.POLYGON, xy,
                                      (np.r_[0, ring_ends].astype(np.int64), np.r_[0, poly_ends].astype(np.int64)))
    if simplify > 0:
        polys = shapely.simplify(polys, simplify, preserve_topology=True)
    if crs:
        polys = to_crs(polys, crs, WEB_MERCATOR)
    return list(zip(polys.tolist(), vals))


EARTH_RADIUS_M = 6_378_137.0


def ground_step(native_res: float, kind: str, lat_deg: float, target_m: float,
                max_upsample: float = 2.0) -> float:
    """Шаг в единицах растра, при котором пиксель равен target_m на земле.

    kind='meter' — UTM, шаг уже в метрах. kind='3857' — веб-меркатор, масштаб
    1/cos(широты). kind='degree' — градусы, 111320·cos(широты) метров на градус.
    Широта — центра растра, не константа одной полосы. Пол 0,2 у косинуса
    срабатывает только при |φ| ≳ 78,5°, это севернее памятников этапа.
    Сданные веса учились в единицах файла: этот шаг включает только кэш с
    pixel_units='ground'.
    """
    native = abs(float(native_res))

    def _cap(native_m: float) -> float:
        return max(float(target_m), native_m / max_upsample)

    if kind == "meter":
        return _cap(native)
    cos = math.cos(math.radians(lat_deg))
    if cos < 0.2:
        cos = 0.2
    meters_per_unit = cos if kind == "3857" else 111_320.0 * cos
    return _cap(native * meters_per_unit) / meters_per_unit


def raster_kind_and_latitude(crs, transform, width: int, height: int) -> tuple[str, float]:
    """('meter'|'3857'|'degree', широта центра). Без CRS считаем единицы метрами."""
    if crs is None:
        return "meter", 0.0
    x, y = transform * (width / 2.0, height / 2.0)
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    if getattr(crs, "is_geographic", False):
        return "degree", float(y)
    if epsg == 3857:
        lat = math.degrees(math.atan(math.sinh(y / EARTH_RADIUS_M)))
        return "3857", lat
    return "meter", 0.0


def _radius_in_ground_meters(crs: str) -> bool:
    """UTM и прочие проекции в метрах. Веб-меркатор и градусы растягивают радиус."""
    from pyproj import CRS

    ref = CRS.from_user_input(crs)
    return not ref.is_geographic and ref.to_epsg() != 3857


def circle_3857(x: float, y: float, radius_m: float, crs: str | None, n: int = 16):
    """Круг радиуса radius_m метров на местности вокруг (x, y) → полигон EPSG:3857.

    В EPSG:3857 и в градусах буфер строится в зоне UTM самой точки. Иначе круг,
    заданный в единицах меркатора, на земле короче в cos φ раз: 4 м дают
    около 2,0 м на 60° с. ш., около 2,2 м на 56° и около 2,7 м на 48°.
    В UTM буфер как раньше: единицы растра уже метры.
    """
    from .crs import transformer, utm_for_lonlat

    src, px, py = crs, float(x), float(y)
    if crs and not _radius_in_ground_meters(crs):
        lon, lat = transformer(crs, "EPSG:4326").transform(px, py)
        src = utm_for_lonlat(float(lon), float(lat))
        px, py = transformer(crs, src).transform(float(x), float(y))
    poly = Point(px, py).buffer(radius_m, quad_segs=max(2, n // 4))
    return to_crs(poly, src, WEB_MERCATOR) if src else poly


_radius_ground_cached = lru_cache(maxsize=256)(_radius_in_ground_meters)   # только для circles_3857


def circles_3857(xs, ys, radius_m: float, crs: str | None, n: int = 16) -> list:
    """circle_3857 для массива точек: те же полигоны побитно, но одним вызовом GEOS/PROJ на зону.

    Порядок операций над каждой точкой тот же (PROJ и GEOS считают точки независимо),
    поэтому координаты совпадают с поштучным вызовом до бита (tests/test_circles_batch.py).
    """
    import shapely

    from .crs import transformer, utm_for_lonlat

    px = np.asarray(xs, dtype=np.float64).ravel()
    py = np.asarray(ys, dtype=np.float64).ravel()
    if px.size == 0:
        return []
    qs = max(2, n // 4)
    if not crs or _radius_ground_cached(crs):
        polys = shapely.buffer(shapely.points(px, py), radius_m, quad_segs=qs)
        if not crs:
            return list(polys)
        return list(_transform_many(polys, crs, WEB_MERCATOR))
    lon, lat = transformer(crs, "EPSG:4326").transform(px, py)
    zones = [utm_for_lonlat(float(a), float(b)) for a, b in zip(np.atleast_1d(lon), np.atleast_1d(lat))]
    out = [None] * px.size
    for z in dict.fromkeys(zones):
        idx = np.array([i for i, q in enumerate(zones) if q == z])
        ux, uy = transformer(crs, z).transform(px[idx], py[idx])
        polys = shapely.buffer(shapely.points(np.asarray(ux, np.float64), np.asarray(uy, np.float64)),
                               radius_m, quad_segs=qs)
        for i, g in zip(idx, _transform_many(polys, z, WEB_MERCATOR)):
            out[i] = g
    return out


def _transform_many(geoms, src: str, dst: str):
    """to_crs для массива геометрий одним вызовом PROJ."""
    import shapely

    from .crs import transformer

    if src == dst:
        return geoms
    tr = transformer(src, dst)
    return shapely.transform(np.asarray(geoms, dtype=object), lambda x, y: tr.transform(x, y), interleaved=False)


def needs_warp(ds) -> str | None:
    """'gcps' или 'geographic', если окна растра нельзя читать как метры.

    Любая проекция в метрах остаётся как есть: зона UTM, национальная сетка
    (поперечная Меркатора на всю страну) и EPSG:3857. Градусы и снимки только
    с опорными точками переводятся в зону UTM своего центра.
    """
    from .crs import is_geographic

    crs = ds.crs.to_string() if ds.crs else None
    if crs is None:
        gcps, gcp_crs = ds.gcps
        return "gcps" if gcps and gcp_crs else None
    return "geographic" if is_geographic(crs) else None


#: Правки чтения снимков (ветка). Секция конфига [raster_fix], по умолчанию пусто = выключено.
#: reproject_3857 — снимок в EPSG:3857 читается перепроецированным в UTM участка (UTM.json):
#:   на трейне 015 Демидовка и 029 Вышегжа_3, шаг файла 1,26 / 1,59 ед. при 0,73 / 0,88 м на земле.
#: int16_identity — int16/uint16 со всеми значениями в 0–255 идёт в сеть как uint8, без растяжки p1–p99
#:   (на трейне 106 Верхмарево растягивался ×3,04).
#: zero_nodata — объект модели, чья внутренняя точка лежит в поле нулей (или nodata) не меньше
#:   zero_nodata_px × zero_nodata_px пикселей окна, выбрасывается (чёрные поля 017 и 036 без nodata).
#: markups — к каким разметкам растров это применяется, по умолчанию только ae.
RASTER_FIX: dict = {}


def configure_raster_fix(spec) -> None:
    """Задаёт [raster_fix] на весь процесс. Кэш band_stats сбрасывается: растяжка от этого зависит."""
    RASTER_FIX.clear()
    if isinstance(spec, dict):
        RASTER_FIX.update(spec)
    try:
        from .tiles import band_stats

        band_stats.cache_clear()
    except Exception:  # pragma: no cover - модуль tiles всегда есть
        pass


def raster_fix_on(key: str, markup: str | None) -> bool:
    """Включена ли правка key для этой разметки (markup None — разметка неизвестна, правка не применяется)."""
    if not RASTER_FIX.get(key) or markup is None:
        return False
    return markup in tuple(RASTER_FIX.get("markups") or ("ae",))


def utm_json_crs(path: str | Path) -> str | None:
    """CRS из UTM.json участка (ищется в папках выше файла): 'urn:ogc:def:crs:EPSG::32636' → 'EPSG:32636'."""
    import json
    import re

    for d in list(Path(path).resolve().parents)[:3]:
        f = d / "UTM.json"
        if f.is_file():
            try:
                m = re.search(r"EPSG:+(\d+)", json.loads(f.read_text(encoding="utf-8-sig")).get("crs", ""))
            except (OSError, ValueError, AttributeError):
                return None
            return f"EPSG:{m.group(1)}" if m else None
    return None


def _reproject_3857(src, path) -> bool:
    from .naming import markup_type_of

    if src.crs is None or not raster_fix_on("reproject_3857", markup_type_of(path)):
        return False
    try:
        return src.crs.to_epsg() == 3857
    except Exception:
        return False


def zero_field_mask(a: np.ndarray, nodata=None, px: int = 5) -> np.ndarray:
    """H×W: True там, где пиксель лежит в поле нулей/nodata всех каналов размером не меньше px×px."""
    from scipy.ndimage import minimum_filter

    z = (a == 0).all(axis=0)
    if nodata is not None:
        z |= (a == nodata).all(axis=0)
    return minimum_filter(z.astype(np.uint8), size=max(1, int(px)), mode="nearest") > 0


@contextmanager
def open_metric(path: str | Path):
    """Растр в метрах. UTM и EPSG:3857 открываются как есть. Градусы и GCP — через WarpedVRT."""
    from rasterio.vrt import WarpedVRT

    from .crs import transformer, utm_for_lonlat

    with rasterio.open(path) as src:
        why = needs_warp(src)
        if why is None and _reproject_3857(src, path):
            # [raster_fix] reproject_3857: веб-меркатор → UTM участка, шаг файла = шаг на земле.
            dst = utm_json_crs(path)
            if dst is None:
                b = src.bounds
                lon, lat = transformer(src.crs.to_string(), "EPSG:4326").transform(
                    (b.left + b.right) / 2, (b.bottom + b.top) / 2)
                dst = utm_for_lonlat(lon, lat)
            log.warning("%s: EPSG:3857 → перепроецирую в %s ([raster_fix] reproject_3857)", Path(path).name, dst)
            with WarpedVRT(src, crs=dst, resampling=Resampling.bilinear) as vrt:
                yield vrt
            return
        if why is None:
            yield src
            return
        if why == "gcps":
            gcps, gcp_crs = src.gcps
            src_crs = gcp_crs
            xs = [g.x for g in gcps]
            ys = [g.y for g in gcps]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        else:
            src_crs = src.crs
            b = src.bounds
            cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        lon, lat = transformer(src_crs.to_string(), "EPSG:4326").transform(cx, cy)
        dst = utm_for_lonlat(lon, lat)
        # Зона — по долготе центра снимка, не по полосе обучения.
        log.warning("%s: %s → перепроецирую в %s", Path(path).name, why, dst)
        with WarpedVRT(src, src_crs=src_crs, crs=dst, resampling=Resampling.bilinear) as vrt:
            yield vrt


def pixel_to_xy(transform, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центры пикселей → координаты CRS растра."""
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return np.asarray(xs), np.asarray(ys)


def meters_to_px(meters: float, res: float) -> int:
    """Длина в метрах → целое число пикселей (не меньше 1)."""
    return max(1, int(math.ceil(meters / res)))
