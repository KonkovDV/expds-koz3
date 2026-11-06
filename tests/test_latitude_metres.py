"""Метры на земле на широтах этапа: Швеция, полоса обучения, Казахстан.

Отбор 12 м и круг заданного радиуса считаются в зоне UTM точки. В единицах
EPSG:3857 тот же отрезок на 60° с. ш. был бы вдвое длиннее, и дубль кургана
прошёл бы мимо порога.
"""
import math

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import Point

from expds.crs import utm_for_lonlat
from expds.pipeline import drop_near_kurgans
from expds.raster import circle_3857

PLACES = ((13.2, 60.0), (37.6, 55.7), (76.4, 48.0))


def _to_3857(lon: float, lat: float) -> tuple[float, float]:
    return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)


def _shift_north(lon: float, lat: float, metres: float) -> tuple[float, float]:
    zone = utm_for_lonlat(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", zone, always_xy=True)
    back = Transformer.from_crs(zone, "EPSG:4326", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    return back.transform(x, y + metres)


def _kurgan(lon: float, lat: float) -> dict:
    x, y = _to_3857(lon, lat)
    return {"region_name": "r", "class_name": "kurgany_tselye", "markup_type": "li",
            "threshold_group": "li", "confidence": 0.9,
            "geom": circle_3857(x, y, 2.0, "EPSG:3857")}


def _ground_radius(poly, lon: float, lat: float) -> float:
    zone = utm_for_lonlat(lon, lat)
    to_utm = Transformer.from_crs("EPSG:3857", zone, always_xy=True).transform
    import shapely

    def _apply(coords):
        xs, ys = to_utm(coords[:, 0], coords[:, 1])
        return np.column_stack([np.asarray(xs), np.asarray(ys)])

    ground = shapely.transform(poly, _apply)
    c = ground.centroid
    dists = [math.hypot(x - c.x, y - c.y) for x, y in list(ground.exterior.coords)[:-1]]
    return max(dists)


@pytest.mark.parametrize("lon,lat", PLACES)
def test_four_metre_circle_stays_four_metres_on_the_ground(lon, lat):
    x, y = _to_3857(lon, lat)
    poly = circle_3857(x, y, 4.0, "EPSG:3857")
    assert _ground_radius(poly, lon, lat) == pytest.approx(4.0, rel=0.04)


@pytest.mark.parametrize("lon,lat", PLACES)
def test_twelve_metre_dedupe_uses_ground_metres(lon, lat):
    anchor = _kurgan(lon, lat)
    near = _kurgan(*_shift_north(lon, lat, 10.0))
    far = _kurgan(*_shift_north(lon, lat, 20.0))
    assert drop_near_kurgans([near], [anchor], 12.0) == []
    assert len(drop_near_kurgans([far], [anchor], 12.0)) == 1


def test_mercator_metres_would_have_kept_the_swedish_duplicate():
    """Контроль смысла: 10 м на земле при 60° — это 20 м EPSG:3857, порог 12 м их не режет."""
    lon, lat = 13.2, 60.0
    anchor = _kurgan(lon, lat)
    near = _kurgan(*_shift_north(lon, lat, 10.0))
    a, b = anchor["geom"].centroid, near["geom"].centroid
    assert math.hypot(a.x - b.x, a.y - b.y) == pytest.approx(20.0, rel=0.05)
    assert Point(a).distance(Point(b)) > 12.0
