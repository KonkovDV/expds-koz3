"""Системы координат: определение и пересчёт в EPSG:3857 для выхода.

Порядок определения CRS региона (как у бейзлайна организаторов):
UTM.json региона → WKT в заголовке LAS → CRS самого растра.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path

import shapely
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)

WEB_MERCATOR = "EPSG:3857"


def las_crs(path: str | Path) -> str | None:
    """CRS из LAS (VLR/EVLR с WKT или GeoTIFF-ключами) через laspy; None, если нет."""
    try:
        import laspy

        with laspy.open(str(path)) as f:
            crs = f.header.parse_crs()
        if crs is None:
            return None
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.to_wkt()
    except Exception as exc:
        log.debug("CRS не прочитан из %s: %s", path, exc)
        return None


def raster_crs(path: str | Path) -> str | None:
    """CRS растра (GeoTIFF) или None."""
    import rasterio

    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        epsg = ds.crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else ds.crs.to_wkt()


@lru_cache(maxsize=64)
def transformer(src: str, dst: str = WEB_MERCATOR) -> Transformer:
    """Кэшируемый трансформер (always_xy: порядок координат x, y)."""
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


def to_crs(geom: BaseGeometry, src: str, dst: str = WEB_MERCATOR) -> BaseGeometry:
    """Перепроецирует геометрию shapely."""
    if src == dst:
        return geom
    tr = transformer(src, dst)
    return shapely.transform(geom, lambda x, y: tr.transform(x, y), interleaved=False)


def utm_for_lonlat(lon: float, lat: float) -> str:
    """EPSG зоны UTM для точки (для метрических расчётов вне растров)."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


@lru_cache(maxsize=256)
def valid_crs(text: str | None) -> str | None:
    """Строка, если pyproj её понимает. «EPSG» без кода — None."""
    if not text:
        return None
    try:
        CRS.from_user_input(text)
    except Exception:
        log.warning("CRS %r не разбирается", text)
        return None
    return text


@lru_cache(maxsize=256)
def is_geographic(crs: str | None) -> bool:
    if not crs:
        return False
    try:
        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:
        return False


def las_box(path: str | Path) -> tuple | None:
    try:
        import laspy

        with laspy.open(str(path)) as f:
            h = f.header
            return float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1])
    except Exception:
        return None


def _box_in(crs_src: str, crs_dst: str, box: tuple) -> tuple | None:
    try:
        tr = transformer(crs_src, crs_dst)
        xs, ys = tr.transform([box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]])
        if not all(map(math.isfinite, list(xs) + list(ys))):
            return None
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return None


def plausible(crs: str, box: tuple | None) -> bool:
    if box is None:
        return True
    ll = _box_in(crs, "EPSG:4326", box)
    if ll is None or not (-180 <= ll[0] <= ll[2] <= 180 and -85 <= ll[1] <= ll[3] <= 85):
        return False
    lon0 = _utm_central_lon(crs)
    return lon0 is None or abs((ll[0] + ll[2]) / 2 - lon0) <= 6.0


def _same(a: str | None, b: str | None) -> bool:
    return _same_crs(a, b)


def _same_crs(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:
        return False


def _overlap_votes(crs: str, box: tuple, rasters: list) -> int:
    import rasterio

    votes = 0
    for path in rasters:
        try:
            with rasterio.open(path) as ds:
                if ds.crs is None:
                    continue
                rb = _box_in(crs, ds.crs.to_string(), box)
                b = ds.bounds
        except Exception:
            continue
        if rb and rb[0] < b.right and rb[2] > b.left and rb[1] < b.top and rb[3] > b.bottom:
            votes += 1
    return votes


def resolve_region_crs(utm_json: str | None, las_files: list, rasters: list, policy: str = "utm_json") -> str | None:
    """CRS участка. По умолчанию UTM.json, как в замере 0,353793.

    policy "las" берёт заголовок LAS. policy "vote" при расхождении выбирает CRS,
    в которой рамка облака ложится на снимки участка; при равенстве остаётся UTM.json.
    """
    utm = valid_crs(utm_json)
    las = valid_crs(las_crs(las_files[0])) if las_files else None
    box = las_box(las_files[0]) if las_files else None
    cands = [c for c in (utm, las) if c]
    if not cands:
        return None
    if len(cands) == 1 or _same_crs(utm, las):
        return cands[0]
    log.warning("CRS расходятся: UTM.json %s, LAS %s", utm, las)
    if policy == "las":
        return las
    if policy != "vote":
        return utm
    ok = [c for c in cands if plausible(c, box)]
    if len(ok) == 1:
        log.warning("выбрана %s: другая CRS не ложится на землю", ok[0])
        return ok[0]
    flat = [p for files in rasters for p in files] if rasters and isinstance(rasters, dict) else list(rasters or [])
    votes = {c: _overlap_votes(c, box, flat) for c in cands} if box and flat else {}
    if votes and votes.get(las, 0) > votes.get(utm, 0):
        log.warning("выбрана CRS LAS %s: растры участка за неё (%s)", las, votes)
        return las
    return utm


def _utm_central_lon(epsg: str | None) -> float | None:
    """Осевой меридиан зоны UTM по коду EPSG:326xx/327xx."""
    if not epsg or not str(epsg).startswith(("EPSG:326", "EPSG:327")):
        return None
    return (int(str(epsg)[-2:]) - 1) * 6 - 180 + 3


def coords_kind(xs: list[float], ys: list[float], fallback_utm: str | None = None) -> str:
    """Вид координат по габаритам: 'wgs84' | 'utm' | 'webmercator' | 'local'.

    Пиксельные координаты (например, профилей электроразведки: x = 120, y = 15)
    легко принять за градусы. Поэтому градусы принимаются, только если точка
    правдоподобна для РФ (широта 35–80°, долгота 15–180°) и, если зона участка
    известна, лежит рядом с её осевым меридианом. Иначе — 'local'.
    """
    if not xs:
        return "empty"
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if -180 <= x0 <= x1 <= 180 and -90 <= y0 <= y1 <= 90:
        lon0 = _utm_central_lon(fallback_utm)
        plausible = 35 <= y0 <= y1 <= 80 and 15 <= x0 <= x1 <= 180
        near = lon0 is None or (lon0 - 6 <= x0 <= x1 <= lon0 + 6)
        return "wgs84" if plausible and near else "local"
    if 100_000 <= x0 <= x1 <= 900_000 and 0 < y0 <= y1 <= 10_000_000:
        return "utm"
    if min(abs(x0), abs(x1)) > 1_000_000 and min(abs(y0), abs(y1)) > 1_000_000:
        return "webmercator"
    return "local"


@lru_cache(maxsize=256)
def to_3857_ok(crs: str | None) -> bool:
    """pyproj переводит эту CRS в EPSG:3857. Безымянная LOCAL_CS и инженерные системы — нет."""
    if not crs:
        return False
    try:
        transformer(crs, WEB_MERCATOR)
    except Exception:
        return False
    return True


# Метры на земле в геофизике (opt-in: [geophys] ground_metres = true или env EXPDS_GROUND_M=1).
# Пороги геофизики (GSD, площади, радиусы, окна признаков) заданы в метрах на земле; в EPSG:3857 единица
# CRS — это cos φ метра (55° с. ш.: 0,57 м). Включено → шаг пикселя растра в веб-меркаторе умножается на
# cos(широты центра растра). Для UTM и прочих метрических CRS множитель 1,0: выход побайтно прежний.
GROUND_METRES: bool | None = None


def ground_metres_on(value=None) -> bool:
    """Аргумент, иначе GROUND_METRES (ставит конвейер из конфига), иначе env EXPDS_GROUND_M."""
    import os

    if value is None:
        value = GROUND_METRES
    if value is None:
        value = os.environ.get("EXPDS_GROUND_M", "0")
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def ground_scale(crs: str | None, bounds=None, enable=None) -> float:
    """Метров на земле в одной единице CRS растра: cos(широты центра) для EPSG:3857, иначе 1,0.
    bounds — (left, bottom, right, top) в этой CRS. Выключено (по умолчанию) — всегда 1,0."""
    if not crs or bounds is None or not ground_metres_on(enable):
        return 1.0
    try:
        if CRS.from_user_input(crs).to_epsg() != 3857:
            return 1.0
    except Exception:
        return 1.0
    y = (float(bounds[1]) + float(bounds[3])) / 2.0
    lat = math.degrees(math.atan(math.sinh(y / 6378137.0)))
    return math.cos(math.radians(lat))


def usable_raster_crs(declared: str | None, bounds, fallback_utm: str | None) -> str | None:
    """CRS растра для перевода в EPSG:3857.

    Объявленная CRS, если pyproj её переводит. Иначе (её нет или это безымянная LOCAL_CS)
    — догадка по габаритам, как у файлов разметки: метры в пределах зоны UTM → UTM.json
    участка, большие метры — веб-меркатор, правдоподобные градусы — WGS 84. Без догадки
    остаётся объявленная: такой растр, как и раньше, падает с ошибкой или выходит в своих
    координатах. Растр с рабочей CRS не меняется.
    """
    if declared and to_3857_ok(declared):
        return declared
    left, bottom, right, top = bounds
    kind = coords_kind([left, right], [min(bottom, top), max(bottom, top)], fallback_utm)
    guess = {"utm": fallback_utm, "webmercator": WEB_MERCATOR, "wgs84": "EPSG:4326"}.get(kind)
    if guess:
        log.warning("CRS растра %s не переводится в EPSG:3857, координаты похожи на %s — беру %s",
                    (declared or "нет")[:40], kind, guess)
        return guess
    return declared


def guess_geojson_crs(data: dict, fallback_utm: str | None) -> str | None:
    """CRS файла разметки: поле "crs", иначе по величине координат; None — координаты локальные.

    None означает, что файл в план не переводится (пиксели профиля, локальная
    система): такой файл пропускается и попадает в отчёт сборки эталона, а не
    превращается молча в объекты где-то в Атлантике.
    """
    crs = data.get("crs")
    if isinstance(crs, dict):
        name = (crs.get("properties") or {}).get("name")
        if name:
            text = str(name)
            if "CRS84" in text.upper():
                return "EPSG:4326"
            if "EPSG::" in text:
                return "EPSG:" + text.split("EPSG::")[-1]
            return text
    xs: list[float] = []
    ys: list[float] = []

    def walk(c):
        if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
            xs.append(float(c[0]))
            ys.append(float(c[1]))
        elif isinstance(c, (list, tuple)):
            for v in c[:50]:
                walk(v)

    for feat in data.get("features", [])[:50]:
        geom = feat.get("geometry") or {}
        walk(geom.get("coordinates", []))
    kind = coords_kind(xs, ys, fallback_utm)
    if kind == "empty":
        return fallback_utm or WEB_MERCATOR
    if kind == "wgs84":
        return "EPSG:4326"
    if kind == "utm":
        return fallback_utm          # None, если UTM.json нет: зону по координатам не угадать
    if kind == "webmercator":
        return WEB_MERCATOR
    return None
