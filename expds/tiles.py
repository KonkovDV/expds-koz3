"""Кэш тайлов для обучения нейросети на слабой видеокарте.

Тайлы нарезаются один раз на CPU и сохраняются в .npy: обучение потом не
ждёт чтения гигантских растров. Позитивы — окна вокруг эталонных объектов
(со сдвигом), негативы — случайные окна и окна вокруг «шума» (extra/*_шум).

Маски хранятся битами: бит k = класс meta["classes"][k] (экономия ×5 по диску).

    python -m expds.tiles --root <данные> --gt work/gt_train.geojson --markup li \
        --out work/cache/li --res 0.5 --tile 256

Для li каналы по умолчанию строятся из LAS (lidar.relief_channels) — той же
функцией, что на инференсе. --li-source tif берёт готовые карты Li трейна;
их нельзя воспроизвести из LAS, поэтому модель на них не переносится на валидацию.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box, shape

from .catalog import Region, read_utm_json, scan_root
from .constants import GEO_CLASSES, GEO_MARKUP, RS_CLASSES
from .crs import WEB_MERCATOR, raster_crs, to_3857_ok, to_crs, usable_raster_crs
from .naming import site_family

log = logging.getLogger(__name__)


def classes_for(markup: str) -> list[str]:
    """Классы, которые учит модель данной модальности."""
    return list(GEO_CLASSES) if markup in GEO_MARKUP else list(RS_CLASSES)


def sources_for(reg: Region, markup: str, li_tags: list[str], exclude: tuple[str, ...] = ()) -> list[list[Path]]:
    """Наборы файлов-каналов для модальности: для li — по тегам (g, c, ch), иначе по файлу.

    exclude — подстроки имён файлов, которые в кэш не идут (карта без сигнала в эталоне
    учит сеть срабатывать на пустом месте; список — scripts/geo_silent_maps.py).
    """
    if markup != "li":
        return [[p] for p in reg.rasters.get(markup, []) if not any(e in p.name for e in exclude)]
    sets: dict[str, dict[str, Path]] = {}
    for p in reg.li_rasters:
        stem, _, tag = p.stem.rpartition("_")
        sets.setdefault(stem, {})[tag] = p
    return [[s[t] for t in li_tags] for s in sets.values() if all(t in s for t in li_tags)]


#: Растр увеличивается до рабочего шага не больше чем в MAX_UPSAMPLE раз: аэрофото 1,6 м
#: на шаге 0,5 м — это 2,4 Гпикс и 44,7 ГБ на одну карту вероятностей.
MAX_UPSAMPLE = 2.0


def working_res(native_res: float, res: float, max_upsample: float = MAX_UPSAMPLE) -> float:
    """Рабочий шаг растра: res, но не мельче native_res / max_upsample."""
    return max(res, native_res / max_upsample)


@lru_cache(maxsize=256)
def band_stats(path: str, markup: str, max_side: int = 2048) -> tuple | None:
    """Перевод каналов растра в uint8, общий для всего снимка: ((a, b), …), y = (x − a)·b; None — уже uint8.

    Считается по уменьшенному обзору снимка, поэтому тайлы обучения и окна
    инференса растягиваются одинаково (по своим перцентилям они растягивались бы по-разному).
    """
    import math

    from .raster import open_metric

    with open_metric(path) as ds:
        dtype = ds.dtypes[0]
        if dtype == "uint8":
            return None
        n = min(ds.count, 3)
        f = max(1, math.ceil(max(ds.width, ds.height) / max_side))
        a = ds.read(list(range(1, n + 1)), out_shape=(n, max(1, ds.height // f), max(1, ds.width // f)),
                    resampling=Resampling.nearest, masked=True).astype(np.float32).filled(np.nan)
    from .raster import raster_fix_on

    if raster_fix_on("int16_identity", markup) and dtype in ("int16", "uint16"):
        fin = a[np.isfinite(a)]
        if fin.size and float(fin.min()) >= 0 and float(fin.max()) <= 255:
            return tuple((0.0, 1.0) for _ in range(a.shape[0]))  # [raster_fix] int16_identity: как uint8
    out = []
    for x in a:
        fin = x[np.isfinite(x)]
        if fin.size == 0:
            out.append((0.0, 1.0))
        elif markup in GEO_MARKUP:
            med = float(np.median(fin))
            mad = float(np.median(np.abs(fin - med))) * 1.4826 or 1.0
            out.append((med - 4 * mad, 31.75 / mad))
        else:
            lo, hi = np.percentile(fin, [1, 99])
            out.append((float(lo), 255.0 / max(float(hi - lo), 1e-6)))
    return tuple(out)


def to_uint8(a: np.ndarray, markup: str, stats: tuple | None = None) -> np.ndarray:
    """Каналы → uint8: геофизика — робастная нормировка, остальное — как есть или растяжка.

    stats (band_stats) задаёт растяжку всего снимка; без них — по самому массиву.
    """
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    out = np.zeros(a.shape, np.uint8)
    if stats is not None:
        for c in range(a.shape[0]):
            lo, k = stats[min(c, len(stats) - 1)]
            out[c] = np.clip(np.nan_to_num((a[c] - lo) * k, nan=0), 0, 255).astype(np.uint8)
        return out
    for c in range(a.shape[0]):
        x = a[c]
        fin = np.isfinite(x)
        if not fin.any():
            continue
        if markup in GEO_MARKUP:
            med = np.median(x[fin])
            mad = np.median(np.abs(x[fin] - med)) * 1.4826 or 1.0
            y = np.clip((x - med) / mad, -4, 4) * 31.75 + 127.5
        else:
            lo, hi = np.percentile(x[fin], [1, 99])
            y = (x - lo) / max(hi - lo, 1e-6) * 255
        out[c] = np.clip(np.nan_to_num(y, nan=0), 0, 255).astype(np.uint8)
    return out


def n_channels(markup: str, li_tags: list[str]) -> int:
    """Число каналов модели: li — по тегам (c, ch — 3, остальные — 1), геофизика — 1, оптика — 3."""
    if markup == "li":
        return sum(3 if t in ("c", "ch") else 1 for t in li_tags)
    return 1 if markup in GEO_MARKUP else 3


def cut_window(arr: np.ndarray, transform, cx: float, cy: float, tile: int):
    """Окно tile×tile из C×H×W, выровненное по сетке массива; снаружи — 0. Возвращает (окно, привязка)."""
    step = transform.a
    col0 = round((cx - transform.c) / step - tile / 2)
    row0 = round((transform.f - cy) / step - tile / 2)
    c, h, w = arr.shape
    out = np.zeros((c, tile, tile), arr.dtype)
    r0, r1 = max(0, row0), min(h, row0 + tile)
    c0, c1 = max(0, col0), min(w, col0 + tile)
    if r1 > r0 and c1 > c0:
        out[:, r0 - row0:r1 - row0, c0 - col0:c1 - col0] = arr[:, r0:r1, c0:c1]
    return out, from_origin(transform.c + col0 * step, transform.f - row0 * step, step, step)


#: Классы, маски которых сжимаются к центру при shrink > 0 (плотные курганные группы).
SHRINK_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye")


def shrink_polygon(g, shrink: float):
    """Полигон, сжатый к центру: радиус эквивалентного круга умножается на (1 − shrink).

    Соседние насыпи на маске перестают касаться, и связные области модели
    разделяют экземпляры. Метрика засчитывает такой объект: покрытие меньшей
    фигуры = 1, внутренняя точка — внутри эталона.
    """
    import math

    if shrink <= 0 or g.geom_type == "Point" or g.area <= 0:
        return g
    r = math.sqrt(g.area / math.pi)
    s = g.buffer(-shrink * r)
    if s.is_empty or s.area < 0.05 * g.area:
        s = g.representative_point().buffer(max(r * (1 - shrink), 0.5))
    return s


def iter_sources(reg: Region, markup: str, li_tags: list[str], li_source: str, res: float, tile: int,
                 dem: dict | None = None, ground_pixels: bool = False, exclude: tuple[str, ...] = ()):
    """Источники тайлов региона: (имя, CRS, рамка, read(cx, cy) → (окно, привязка))."""
    if markup == "li" and li_source == "las":
        from .lidar import MODEL_DEM, build_dem, relief_channels

        dem = dem or MODEL_DEM
        for las in reg.las:
            keep = float(dem.get("keep_prob", 1.0))
            if dem.get("thin"):
                from .lidar import thin_keep
                keep = thin_keep(las.name, int(dem.get("thin_seed", 0)))
                log.info("прореживание %s: оставляю %.3f точек земли", las.name, keep)
            d = build_dem(las, res=dem["res"], fill=dem["fill"], crs=read_utm_json(reg.utm_json),
                          keep_prob=keep)
            if d.crs is None:
                log.warning("%s: нет CRS у %s — пропуск", reg.name, las.name)
                continue
            arr, tr = relief_channels(d, res, tuple(li_tags))
            h, w = arr.shape[1:]
            bnd = box(tr.c, tr.f - h * tr.a, tr.c + w * tr.a, tr.f)
            yield las.name, d.crs, bnd, lambda cx, cy, a=arr, t=tr: cut_window(a, t, cx, cy, tile)
        return
    for files in sources_for(reg, markup, li_tags, exclude):
        utm = read_utm_json(reg.utm_json)
        from .raster import open_metric

        with open_metric(files[0]) as ds:
            # Градусная сетка уже в UTM. Безымянная LOCAL_CS с метрами зоны получает UTM.json.
            declared = ds.crs.to_string() if ds.crs else None
            crs = usable_raster_crs(declared, ds.bounds, utm) or utm
            if crs is None:
                log.warning("%s: нет CRS у %s — пропуск", reg.name, files[0].name)
                continue
            bnd = box(*ds.bounds)
            native = abs(ds.res[0])
            if ground_pixels:
                from rasterio.crs import CRS as RioCRS

                from .raster import ground_step, raster_kind_and_latitude

                kind_crs = ds.crs if to_3857_ok(declared) else (RioCRS.from_user_input(crs) if crs else None)
                kind, lat = raster_kind_and_latitude(kind_crs, ds.transform, ds.width, ds.height)
                step = ground_step(native, kind, lat, res)
            else:
                step = working_res(native, res)
            stats = band_stats(str(files[0]), markup) if len(files) == 1 else None
            if len(files) > 1:
                # Каналы лежат в разных файлах (карты Li трейна): каждый файл открывается заново.
                yield files[0].name, crs, bnd, lambda cx, cy, f=files, r=step, st=stats: (
                    to_uint8(read_window(f, cx, cy, r, tile), markup, st),
                    from_origin(cx - tile * r / 2, cy + tile * r / 2, r, r))
                continue
            # Файл остаётся открытым, пока вызывающий читает окна этого источника.
            # Иначе каждый тайл заново строит WarpedVRT большого ортофото.
            strips = StripGrid(ds, step) if StripGrid.applies(ds) else None
            log.info("%s: источник %s%s", reg.name, files[0].name, ", полосы в рабочем шаге" if strips else "")

            def read_open(cx, cy, ds=ds, r=step, st=stats, strips=strips):
                if strips is not None:
                    img, x0, y0 = strips.window(cx, cy, tile)
                    return to_uint8(img, markup, st), from_origin(x0, y0, strips.step_x, strips.step_y)
                return (to_uint8(window_from_dataset(ds, cx, cy, r, tile), markup, st),
                        from_origin(cx - tile * r / 2, cy + tile * r / 2, r, r))

            yield files[0].name, crs, bnd, read_open


def fix_channels(img: np.ndarray, c: int) -> np.ndarray:
    """Приводит число каналов к c: лишние отбрасываются, недостающие повторяют последний."""
    if img.shape[0] >= c:
        return img[:c]
    from . import speed

    if not speed.on("fast_channels"):
        return np.concatenate([img, np.repeat(img[-1:], c - img.shape[0], axis=0)], axis=0)
    out = np.empty((c, *img.shape[1:]), img.dtype)     # флаг fast_channels: без repeat + concatenate
    out[:img.shape[0]] = img
    out[img.shape[0]:] = img[-1]
    return out


def _read_resampled(ds, win: Window, n: int, tile: int, resampling) -> np.ndarray:
    """Окно в кадр tile×tile. За краем — нули.

    Обычный растр читается с boundless. WarpedVRT так не умеет: читается только
    перекрытие с кадром, остальное остаётся нулём. Пиксели внутри кадра те же.
    """
    bands = list(range(1, n + 1))
    try:
        return ds.read(bands, window=win, out_shape=(n, tile, tile), boundless=True,
                       fill_value=0, resampling=resampling)
    except ValueError:
        # Тип пикселя снимка: uint8 без растяжки идёт в сеть как есть, как на инференсе.
        out = np.zeros((n, tile, tile), ds.dtypes[0])
        col0, row0 = float(win.col_off), float(win.row_off)
        width, height = float(win.width), float(win.height)
        if width <= 0 or height <= 0:
            return out
        left, top = max(col0, 0.0), max(row0, 0.0)
        right = min(col0 + width, float(ds.width))
        bottom = min(row0 + height, float(ds.height))
        if right <= left or bottom <= top:
            return out
        c0 = max(0, int(math.floor((left - col0) / width * tile)))
        c1 = min(tile, int(math.ceil((right - col0) / width * tile)))
        r0 = max(0, int(math.floor((top - row0) / height * tile)))
        r1 = min(tile, int(math.ceil((bottom - row0) / height * tile)))
        if c1 <= c0 or r1 <= r0:
            return out
        part = ds.read(bands, window=Window(left, top, right - left, bottom - top),
                       out_shape=(n, r1 - r0, c1 - c0), resampling=resampling)
        out[:, r0:r1, c0:c1] = part
        return out


class StripGrid:
    """Полосатый снимок, прочитанный на всю ширину сразу в рабочем шаге.

    Строка такого файла — отдельный блок на всю ширину, уменьшенных копий нет.
    Окно 128 м на шаге 7 мм — это 17 500 исходных строк: читать их на каждое окно
    заново — часы на файл. Здесь строки рабочей сетки читаются один раз, со
    средним по исходным пикселям, как у сдачи (`Resampling.average`). Окно
    привязано к этой сетке: сдвиг центра не больше половины рабочего пикселя,
    маска строится по той же привязке.
    """

    def __init__(self, ds, step: float):
        t = ds.transform
        self.ds = ds
        self.bands = list(range(1, min(ds.count, 3) + 1))
        self.scale_x = step / abs(t.a)
        self.scale_y = step / abs(t.e)
        self.step_x = abs(t.a) * self.scale_x
        self.step_y = abs(t.e) * self.scale_y
        self.width = max(1, int(ds.width / self.scale_x))
        self.height = max(1, int(ds.height / self.scale_y))
        self.dtype = ds.dtypes[0]
        self.rows: dict[int, np.ndarray] = {}

    @staticmethod
    def applies(ds) -> bool:
        """Обычный файл без поворота, строка — блок на всю ширину, без уменьшенных копий."""
        from rasterio.vrt import WarpedVRT

        if isinstance(ds, WarpedVRT):
            return False
        t = ds.transform
        if t.b != 0 or t.d != 0 or t.e >= 0:
            return False
        try:
            block_h, block_w = ds.block_shapes[0]
            overviews = ds.overviews(1)
        except Exception:
            return False
        return block_h == 1 and block_w >= ds.width and not overviews

    def _read_rows(self, start: int, end: int) -> None:
        start, end = max(start, 0), min(end, self.height)
        if end <= start:
            return
        native = Window(0, start * self.scale_y, self.width * self.scale_x, (end - start) * self.scale_y)
        block = self.ds.read(self.bands, window=native, out_shape=(len(self.bands), end - start, self.width),
                             resampling=Resampling.average)
        for i, row in enumerate(range(start, end)):
            self.rows[row] = block[:, i, :]

    def window(self, cx: float, cy: float, tile: int) -> tuple[np.ndarray, float, float]:
        """Окно tile×tile и левый верхний угол его привязки."""
        t = self.ds.transform
        col = (cx - t.c) / (t.a * self.scale_x)
        row = (cy - t.f) / (t.e * self.scale_y)
        j0 = int(round(col - tile / 2))
        i0 = int(round(row - tile / 2))
        # Держим только строки текущего окна: память — одна полоса окна на всю ширину.
        self.rows = {k: v for k, v in self.rows.items() if i0 <= k < i0 + tile}
        missing = [k for k in range(i0, i0 + tile) if k not in self.rows and 0 <= k < self.height]
        if missing:
            self._read_rows(missing[0], missing[-1] + 1)
        out = np.zeros((len(self.bands), tile, tile), self.dtype)
        c0, c1 = max(j0, 0), min(j0 + tile, self.width)
        if c1 > c0:
            for i, k in enumerate(range(i0, i0 + tile)):
                line = self.rows.get(k)
                if line is not None:
                    out[:, i, c0 - j0:c1 - j0] = line[:, c0:c1]
        x0 = t.c + j0 * self.scale_x * t.a
        y0 = t.f + i0 * self.scale_y * t.e
        return out, x0, y0


def window_from_dataset(ds, cx: float, cy: float, res: float, tile: int) -> np.ndarray:
    """Окно tile×tile уже открытого растра. За краем — нули."""
    half = tile * res / 2
    inv = ~ds.transform
    x0, y0, x1, y1 = cx - half, cy + half, cx + half, cy - half
    col0, row0 = inv.a * x0 + inv.b * y0 + inv.c, inv.d * x0 + inv.e * y0 + inv.f
    col1, row1 = inv.a * x1 + inv.b * y1 + inv.c, inv.d * x1 + inv.e * y1 + inv.f
    win = Window(col0, row0, col1 - col0, row1 - row0)
    n = min(ds.count, 3)
    scale = res / abs(ds.res[0])
    resampling = Resampling.average if scale > 1 else Resampling.bilinear
    return _read_resampled(ds, win, n, tile, resampling)


def read_window(files: list[Path], cx: float, cy: float, res: float, tile: int) -> np.ndarray:
    """Окно tile×tile с центром (cx, cy) в CRS растра и шагом res; все каналы файлов подряд.

    Градусный снимок читается уже в UTM, как на инференсе.
    """
    from .raster import open_metric

    chans = []
    for f in files:
        with open_metric(f) as ds:
            chans.append(window_from_dataset(ds, cx, cy, res, tile))
    return np.concatenate(chans, axis=0)


def ground_area_m2(g, crs: str) -> float:
    """Площадь полигона в квадратных метрах на земле.

    В UTM это площадь фигуры. В веб-меркаторе и в градусах фигура сначала
    переводится в зону UTM центроида: иначе на широте Москвы квадратный метр
    карты примерно втрое больше метра на земле.
    """
    if g.is_empty or g.geom_type == "Point":
        return 0.0
    from .crs import transformer, utm_for_lonlat
    from .raster import _radius_in_ground_meters

    if _radius_in_ground_meters(crs):
        return float(g.area)
    c = g.centroid
    lon, lat = transformer(crs, "EPSG:4326").transform(c.x, c.y)
    utm = utm_for_lonlat(float(lon), float(lat))
    return float(to_crs(g, crs, utm).area)


def shift_labels(local: list[tuple], shift_m: tuple[float, float], crs: str, bnd) -> list[tuple]:
    """Перенос эталона снимка на (восток, юг) метров на земле.

    В UTM единицы уже метры. В веб-меркаторе метр на земле длиннее единицы карты.
    В градусах перенос идёт через зону UTM: пять единиц CRS здесь были бы пятью градусами.
    """
    import math

    from shapely import affinity

    from .crs import is_geographic, to_crs, utm_for_lonlat

    east, south = float(shift_m[0]), float(shift_m[1])
    if is_geographic(crs):
        c = bnd.centroid
        utm = utm_for_lonlat(float(c.x), float(c.y))
        out = []
        for geom, cls in local:
            moved = affinity.translate(to_crs(geom, crs, utm), east, -south)
            out.append((to_crs(moved, utm, crs), cls))
        return out
    unit = 1.0
    if str(crs).upper().endswith(":3857"):
        lat = math.degrees(math.atan(math.sinh(bnd.centroid.y / 6378137.0)))
        unit = 1.0 / max(math.cos(math.radians(lat)), 0.2)
    dx, dy = east * unit, -south * unit
    return [(affinity.translate(g, dx, dy), c) for g, c in local]


def load_label_shifts(path: str | None) -> dict[str, tuple[float, float]]:
    """Устойчивые сдвиги меток по файлам снимков из JSON scripts/image_label_shift.py --out."""
    if not path:
        return {}
    rows = json.loads(Path(path).read_text(encoding="utf-8")).get("rasters", [])
    return {r["file"]: (float(r["dx_m"]), float(r["dy_m"])) for r in rows if r.get("systematic")}


#: Бит маски «не учить»: классов в кэше не больше семи, старший бит свободен.
IGNORE_BIT = 7
CLASS_BITS_MASK = (1 << IGNORE_BIT) - 1


def grid_centers(minx: float, miny: float, maxx: float, maxy: float, span: float, step: float) -> list[tuple[float, float]]:
    """Центры окон, как у инференса: первое встык с ближним краем, последнее — с дальним.

    `infer_nn.Segmenter.predict` ставит окно в пиксель 0 и дописывает окно встык с концом
    кадра, если шаг его не достал. Здесь то же в метрах источника. Окно, чей центр лежит
    на дальнем крае, наполовину выходит за источник — инференс так не режет.
    """
    def starts(length: float) -> list[float]:
        if length <= span + 1e-6:
            return [0.0]
        out = list(np.arange(0.0, max(length - span, 0.0) + 1e-6, step))
        if not out:
            out = [0.0]
        if out[-1] + span < length - 1e-6:
            out.append(length - span)
        return out

    half = span / 2.0
    return [(minx + sx + half, maxy - sy - half)
            for sy in starts(maxy - miny) for sx in starts(maxx - minx)]


def build_cache(root: str, gt_path: str, markup: str, out: str, res: float = 0.5, tile: int = 256,
                neg_per_pos: float = 1.0, jitter: float = 0.25, li_tags: list[str] | None = None,
                max_per_region: int = 3000, seed: int = 0, shard_size: int = 2000,
                li_source: str = "las", shrink: float = 0.0, dem: dict | None = None,
                shrink_classes: tuple[str, ...] | None = None,
                ground_pixels: bool = False,
                only_classes: list[str] | None = None, pos_repeat: int = 1,
                only_regions: list[str] | None = None, min_area_m2: float = 0.0,
                exclude_files: list[str] | None = None,
                label_shifts: dict[str, tuple[float, float]] | None = None,
                ignore_other_layers: bool = False, min_clip_frac: float = 0.0,
                grid_overlap: float | None = None) -> dict:
    """Строит кэш тайлов и возвращает сводку.

    dem — {res, fill} DEM для li из LAS (по умолчанию MODEL_DEM); only_classes — учить
    только эти классы (например, городища на грубом шаге); pos_repeat — тайлов на
    объект (каждый со своим случайным сдвигом), для редких классов.
    min_area_m2 — не брать полигоны мельче этого размера на земле; точки остаются.
    label_shifts — {имя файла: (на восток, на юг) метров}: эталон этого снимка переносится
    на пятна самого снимка; снимков без записи перенос не касается.
    ignore_other_layers — объект тех же классов из разметки другого слоя там, где своей
    разметки нет, пишется битом IGNORE_BIT: «не учить», а не фон. Обучение читает бит
    только с флагом (`expds.train`, `use_ignore`).
    min_clip_frac — объект, от которого в окне меньше этой доли площади, пишется тем же битом.
    grid_overlap — вместо выборки вокруг объектов сплошная сетка окон по всему источнику с этим
    перекрытием (0–0,9), как на инференсе: кэш для честной оценки детектора рамок на целых
    участках (`scripts/coco_from_cache.py`). Пустые окна (вне съёмки) пропускаются.
    В meta.json пишется CRS каждого источника (`crs_by_source`): рамку из окна по нему
    возвращают на землю.
    """
    from .lidar import MODEL_DEM, RELIEF_TAGS

    dem = dict(dem or MODEL_DEM)
    rng = random.Random(seed)
    li_tags = list(li_tags or (RELIEF_TAGS if li_source == "las" else ["g"]))
    n_ch = n_channels(markup, li_tags)
    classes = [c for c in classes_for(markup) if not only_classes or c in only_classes]
    if not classes:
        raise ValueError(f"нет классов {only_classes} в модальности {markup}")
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))["features"]
    if (ignore_other_layers or min_clip_frac > 0) and len(classes) > IGNORE_BIT:
        raise ValueError(f"бит «не учить» {IGNORE_BIT} занят: классов {len(classes)}")
    by_region: dict[str, list[tuple]] = {}
    others: dict[str, list] = {}
    for f in gt:
        pr = f["properties"]
        if pr.get("markup_type") == markup and pr["class_name"] in classes:
            by_region.setdefault(pr["region_name"], []).append((shape(f["geometry"]), pr["class_name"]))
        elif ignore_other_layers and pr["class_name"] in classes:
            others.setdefault(pr["region_name"], []).append(shape(f["geometry"]))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs, masks, rows = [], [], []
    crs_by_source: dict[str, str] = {}
    shard = 0
    total = {"tiles": 0, "positive": 0}

    def flush():
        nonlocal shard, imgs, masks
        if not imgs:
            return
        np.save(out_dir / f"images_{shard:04d}.npy", np.stack(imgs))
        np.save(out_dir / f"masks_{shard:04d}.npy", np.stack(masks))
        shard += 1
        imgs, masks = [], []

    for reg in scan_root(root):
        if only_regions is not None and reg.name not in only_regions:
            continue
        objs = by_region.get(reg.name, [])
        for src_name, crs, bnd, read in iter_sources(reg, markup, li_tags, li_source, res, tile, dem,
                                                     ground_pixels=ground_pixels,
                                                     exclude=tuple(exclude_files or ())):
            crs_by_source[f"{reg.name}|{src_name}"] = str(crs)
            local = [(to_crs(g, WEB_MERCATOR, crs), c) for g, c in objs]
            if label_shifts and src_name in label_shifts:
                local = shift_labels(local, label_shifts[src_name], crs, bnd)
            local = [(g, c) for g, c in local if g.intersects(bnd)]
            foreign = [g.buffer(1.0) if g.geom_type == "Point" else g
                       for g in (to_crs(g, WEB_MERCATOR, crs) for g in others.get(reg.name, [])) if g.intersects(bnd)]
            if min_area_m2 > 0:
                local = [(g, c) for g, c in local
                         if g.geom_type == "Point" or ground_area_m2(g, crs) >= min_area_m2]
            minx, miny, maxx, maxy = bnd.bounds
            if grid_overlap is not None:
                step = tile * res * (1.0 - min(max(grid_overlap, 0.0), 0.9))
                centers = [(x, y, 0) for x, y in grid_centers(minx, miny, maxx, maxy, tile * res, step)]
            else:
                centers = [(g.representative_point().x, g.representative_point().y, 1) for g, _ in local] * pos_repeat
                rng.shuffle(centers)
                centers = centers[:max_per_region]
                n_neg = max(5, int(len(centers) * neg_per_pos))
                centers += [(rng.uniform(minx, maxx), rng.uniform(miny, maxy), 0) for _ in range(n_neg)]
            for noise in (reg.noise if grid_overlap is None else []):
                try:
                    data = json.loads(noise.read_text(encoding="utf-8"))
                    for nf in data.get("features", [])[:200]:
                        p = to_crs(shape(nf["geometry"]), WEB_MERCATOR, crs).representative_point()
                        if bnd.contains(p):
                            centers.append((p.x, p.y, 0))
                except Exception:
                    pass
            ordered = []
            for cx, cy, pos in centers:
                j = tile * res * jitter
                if pos:
                    cx += rng.uniform(-j, j)
                    cy += rng.uniform(-j, j)
                ordered.append((cy, cx, pos))
            # С севера на юг: строки снимка идут сверху вниз, полоса не перечитывается.
            ordered.sort(key=lambda item: (-item[0], item[1]))
            for cy, cx, pos in ordered:
                img, tr = read(cx, cy)
                if not img.any():
                    continue
                m = np.zeros((tile, tile), np.uint8)
                if min_clip_frac > 0:
                    from rasterio.transform import array_bounds
                    from shapely.geometry import box as shape_box

                    from .open_ideas import split_clipped

                    west, south, east, north = array_bounds(tile, tile, tr)
                    window = shape_box(west, south, east, north)
                for k, cls in enumerate(classes):
                    shp = [(g.buffer(1.0) if g.geom_type == "Point"
                            else shrink_polygon(g, shrink) if cls in (shrink_classes or SHRINK_CLASSES) else g)
                           for g, c in local if c == cls]
                    if min_clip_frac > 0 and shp:
                        shp, skipped = split_clipped(shp, window, min_clip_frac)
                        if skipped:
                            r = rasterize(((g, 1) for g in skipped), out_shape=(tile, tile), transform=tr,
                                          all_touched=True, dtype="uint8")
                            m |= ((r.astype(bool) & (m == 0)).astype(np.uint8) << IGNORE_BIT)
                    if shp:
                        r = rasterize(((g, 1) for g in shp), out_shape=(tile, tile), transform=tr,
                                      all_touched=True, dtype="uint8")
                        m |= (r.astype(np.uint8) << k)
                if foreign:
                    r = rasterize(((g, 1) for g in foreign), out_shape=(tile, tile), transform=tr,
                                  all_touched=True, dtype="uint8")
                    m |= ((r.astype(bool) & (m == 0)).astype(np.uint8) << IGNORE_BIT)
                imgs.append(fix_channels(to_uint8(img, markup), n_ch))
                masks.append(m)
                own = bool((m & CLASS_BITS_MASK).any())
                rows.append([shard, len(imgs) - 1, reg.name, site_family(reg.name), src_name,
                             round(cx, 2), round(cy, 2), int(own)])
                total["tiles"] += 1
                total["positive"] += int(own)
                if len(imgs) >= shard_size:
                    flush()
        log.info("%s: тайлов всего %d", reg.name, total["tiles"])
    flush()
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shard", "idx", "region", "family", "file", "cx", "cy", "positive"])
        w.writerows(rows)
    li_src = li_source if markup == "li" else None
    meta = {"markup": markup, "classes": classes, "res": res, "tile": tile,
            "li_tags": li_tags if markup == "li" else None,
            "dem_res": dem["res"] if li_src == "las" else None, "dem_fill": dem["fill"] if li_src == "las" else None,
            "li_source": li_src, "flips": markup != "li" or (li_src == "las" and "g" not in li_tags),
            "channels": n_ch, "shrink": shrink, "shrink_classes": list(shrink_classes or SHRINK_CLASSES), "max_upsample": MAX_UPSAMPLE,
            "pixel_units": "ground" if ground_pixels else "native",
            "min_area_m2": min_area_m2, "exclude_files": list(exclude_files or []), **total}
    if label_shifts:
        meta["label_shifts"] = {k: list(v) for k, v in label_shifts.items()}
    if ignore_other_layers or min_clip_frac > 0:
        meta["ignore_bit"] = IGNORE_BIT
    if min_clip_frac > 0:
        meta["min_clip_frac"] = min_clip_frac
    if grid_overlap is not None:
        meta["grid_overlap"] = grid_overlap
    meta["crs_by_source"] = crs_by_source
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Кэш тайлов для обучения")
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--markup", required=True, choices=["li", "ae", "sp_or", "or", "maggr", "etgr", "terfl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--li-source", choices=["las", "tif"], default="las",
                    help="li: каналы из LAS (как на инференсе) или готовые карты Li трейна")
    ap.add_argument("--li-tags", nargs="*", default=None,
                    help="las: lrm_s lrm_l lrm_xl slope g dev_s dev_l dev_xl rough curv int opns svf; tif: g i c ch")
    ap.add_argument("--max-per-region", type=int, default=3000)
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="сжать маски курганов к центру на эту долю радиуса (разделение плотных групп)")
    ap.add_argument("--shrink-classes", nargs="*", default=None,
                    help="какие классы сжимать (по умолчанию курганы). Архитектура: касающиеся комнаты и "
                         "стены иначе сливаются в одну компоненту, и цель карты центров ставит один пик "
                         "в центре масс слитой фигуры, часто вне эталона")
    ap.add_argument("--dem-res", type=float, default=None, help="шаг DEM для li из LAS (по умолчанию 0,5 м)")
    ap.add_argument("--dem-fill", choices=["pyramid", "delaunay"], default=None)
    ap.add_argument("--thin", action="store_true",
                    help="в половине облаков оставить 25–100 %% точек земли до сетки")
    ap.add_argument("--classes", nargs="*", default=None, help="учить только эти классы")
    ap.add_argument("--pos-repeat", type=int, default=1, help="тайлов на объект (разные сдвиги)")
    ap.add_argument("--ground-pixels", action="store_true",
                    help="шаг тайла в метрах на земле: веб-меркатор делится на cos(широты)")
    ap.add_argument("--regions", nargs="*", default=None, help="только эти участки")
    ap.add_argument("--label-shifts", default=None,
                    help="JSON scripts/image_label_shift.py: эталон снимка переносится на его пятна (только устойчивые)")
    ap.add_argument("--exclude-files", nargs="*", default=None,
                    help="подстроки имён растров, которые в кэш не идут (scripts/geo_silent_maps.py)")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="не брать полигоны мельче этого размера, м² на земле; точки остаются")
    ap.add_argument("--ignore-other-layers", action="store_true",
                    help="объекты тех же классов из разметки других слоёв — бит «не учить», а не фон")
    ap.add_argument("--clip-ignore", type=float, default=0.0,
                    help="доля площади объекта в окне, ниже которой пиксели — «не учить», а не фон; 0 — как раньше")
    ap.add_argument("--grid-overlap", type=float, default=None,
                    help="сплошная сетка окон с этим перекрытием вместо выборки вокруг объектов (оценка детектора)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from .lidar import MODEL_DEM

    dem = {"res": args.dem_res or MODEL_DEM["res"], "fill": args.dem_fill or MODEL_DEM["fill"],
           "thin": args.thin}
    meta = build_cache(args.root, args.gt, args.markup, args.out, args.res, args.tile, args.neg_per_pos,
                       li_tags=args.li_tags, max_per_region=args.max_per_region, li_source=args.li_source,
                       shrink=args.shrink, dem=dem, ground_pixels=args.ground_pixels,
                       shrink_classes=tuple(args.shrink_classes) if args.shrink_classes else None,
                       only_classes=args.classes, pos_repeat=args.pos_repeat,
                       only_regions=args.regions, min_area_m2=args.min_area, exclude_files=args.exclude_files,
                       label_shifts=load_label_shifts(args.label_shifts),
                       ignore_other_layers=args.ignore_other_layers,
                       min_clip_frac=args.clip_ignore, grid_overlap=args.grid_overlap)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
