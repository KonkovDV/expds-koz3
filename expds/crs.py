"""Системы координат: определение и пересчёт в EPSG:3857 для выхода.

Порядок определения CRS региона (как у бейзлайна организаторов):
UTM.json региона → WKT в заголовке LAS → CRS самого растра.
"""
from __future__ import annotations

import logging
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
