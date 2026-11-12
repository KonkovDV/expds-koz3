"""Физика каналов и математика балла против официальной метрики (аудит 12.11).

Все проверки на синтетике. Официальный файл — materials/compute_metrics.py.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from pyproj import Geod, Transformer
from shapely.geometry import Point, box, mapping

from expds import accel
from expds.post import _ground_scale
from expds.raster import circle_3857

ROOT = Path(__file__).resolve().parents[1]
UTM36 = Transformer.from_crs(32636, 3857, always_xy=True)
X0, Y0 = 500_000.0, 6_100_000.0


def _official():
    spec = importlib.util.spec_from_file_location("koz3_official_metric", ROOT / "materials" / "compute_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CM = _official()


def _poly_3857(g_utm):
    import shapely

    return mapping(shapely.transform(g_utm, lambda x, y: UTM36.transform(x, y), interleaved=False))


def _feat(g_utm, cls, conf=1.0, region="region_0001", markup=None):
    props = {"class_name": cls, "region_name": region, "confidence": conf}
    if markup:
        props["markup_type"] = markup
    return {"type": "Feature", "geometry": _poly_3857(g_utm), "properties": props}


def _fc(feats):
    return {"type": "FeatureCollection", "features": feats}


def _case(gt_n: int, hits: int, misses: int, cls: str = "kurgany_tselye"):
    """gt_n кругов эталона, hits предсказаний на них и misses далеко от любого эталона."""
    gt = [_feat(Point(X0 + 30 * i, Y0).buffer(5), cls) for i in range(gt_n)]
    pr = [_feat(Point(X0 + 30 * i, Y0).buffer(5), cls, 0.9) for i in range(hits)]
    pr += [_feat(Point(X0 + 30 * i, Y0 + 5_000).buffer(5), cls, 0.5) for i in range(misses)]
    return pr, gt


def _f1(pr, gt, cls="kurgany_tselye"):
    return CM.F1Calculator().calculate_f1(_fc(pr), _fc(gt))["per_class"][cls]["f1"]


@pytest.mark.parametrize("gt_n,t,f", [(40, 10, 30), (60, 30, 5), (25, 20, 40), (80, 8, 120)])
def test_class_f1_is_2t_over_g_plus_t_plus_half_fp(gt_n, t, f):
    pr, gt = _case(gt_n, t, f)
    assert _f1(pr, gt) == pytest.approx(2 * t / (gt_n + t + 0.5 * f), abs=1e-5)


def test_break_even_hit_rate_is_f_over_four_minus_f():
    gt_n, t, f = 100, 30, 40
    base = 2 * t / (gt_n + t + 0.5 * f)
    p_star = base / (4 - base)
    k = 200
    above, below = math.ceil(k * p_star) + 2, math.floor(k * p_star) - 2
    pr_up, gt = _case(gt_n, t + above, f + k - above)
    pr_dn, _ = _case(gt_n, t + below, f + k - below)
    assert _f1(pr_up, gt) > base > _f1(pr_dn, gt)


def test_class_scores_are_independent_so_class_disjoint_changes_add_up():
    kur, finds = "kurgany_povrezhdennye", "finds_points"
    gt = [_feat(Point(X0 + 30 * i, Y0).buffer(5), kur) for i in range(10)]
    gt += [_feat(Point(X0 + 30 * i, Y0 + 900).buffer(5), finds) for i in range(10)]
    kur_a = [_feat(Point(X0 + 30 * i, Y0).buffer(5), kur) for i in range(3)]
    kur_b = [_feat(Point(X0 + 30 * i, Y0).buffer(5), kur) for i in range(7)]
    geo_a = [_feat(Point(X0 + 30 * i, Y0 + 900).buffer(5), finds) for i in range(8)]
    geo_b = [_feat(Point(X0 + 30 * i, Y0 + 900).buffer(5), finds) for i in range(2)]

    def score(pr):
        return CM.F1Calculator().calculate_f1(_fc(pr), _fc(gt))["final_score"]

    base, lidar, geo, both = score(kur_a + geo_a), score(kur_b + geo_a), score(kur_a + geo_b), score(kur_b + geo_b)
    assert both == pytest.approx(geo + (lidar - base), abs=1e-12)


def test_matching_pools_regions_and_ignores_markup_type():
    cls = "kurgany_povrezhdennye"
    gt = [_feat(Point(X0, Y0).buffer(5), cls, region="region_0001", markup="li")]
    pr = [_feat(Point(X0, Y0).buffer(5), cls, region="region_9999", markup="sp_or")]
    res = CM.F1Calculator().calculate_f1(_fc(pr), _fc(gt))
    assert res["per_class"][cls]["tp"] == 1


def test_micro_circle_hits_where_a_two_metre_circle_on_a_thin_wall_misses():
    cls = "fortifikatsii"
    wall = box(X0 - 10, Y0 - 0.5, X0 + 10, Y0 + 0.5)
    gt = [_feat(wall, cls)]
    big = [_feat(Point(X0, Y0).buffer(2.0), cls)]
    tiny = [_feat(Point(X0, Y0).buffer(0.25), cls)]
    calc = CM.F1Calculator()
    assert calc.calculate_f1(_fc(big), _fc(gt))["per_class"][cls]["tp"] == 0
    assert calc.calculate_f1(_fc(tiny), _fc(gt))["per_class"][cls]["tp"] == 1


@pytest.mark.parametrize("lat", [48.0, 56.0, 60.0, 64.0])
def test_ground_scale_matches_ellipsoid_parallel(lat):
    tr = Transformer.from_crs(4326, 3857, always_xy=True)
    x, y = tr.transform(30.0, lat)
    dx = 100.0
    lon2, _ = Transformer.from_crs(3857, 4326, always_xy=True).transform(x + dx, y)
    _, _, ground = Geod(ellps="WGS84").inv(30.0, lat, lon2, lat)
    assert _ground_scale(y) == pytest.approx(math.cos(math.radians(lat)), rel=1e-9)
    assert ground / dx == pytest.approx(_ground_scale(y), rel=3e-3)


@pytest.mark.parametrize("lat", [48.0, 60.0])
def test_four_metre_circle_is_four_metres_in_the_metric_frame(lat):
    lon = 15.5
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
    poly = circle_3857(x, y, 4.0, "EPSG:3857")
    utm = CM._to_utm(poly, "webmercator")
    c = utm.centroid
    r = [math.hypot(px - c.x, py - c.y) for px, py in list(utm.exterior.coords)[:-1]]
    assert max(r) == pytest.approx(4.0, abs=2e-3)
    assert min(r) == pytest.approx(4.0, abs=2e-3)


def _plane(k_e: float, k_s: float, n: int = 21) -> np.ndarray:
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float32)
    return (k_e * cols + k_s * rows).astype(np.float32)


def test_horn_slope_of_a_plane():
    z = _plane(0.1, 0.0)
    s = accel.slope_deg(z, 1.0)
    assert s[5:-5, 5:-5] == pytest.approx(math.degrees(math.atan(0.1)), abs=1e-3)


def test_hillshade_lights_from_the_northwest_on_a_north_up_grid():
    k = 1 / math.sqrt(2)
    lit = accel.hillshade(_plane(k, k), 1.0, 1.0)
    dark = accel.hillshade(_plane(-k, -k), 1.0, 1.0)
    assert int(np.median(lit[3:-3, 3:-3])) >= 254
    assert int(np.median(dark[3:-3, 3:-3])) <= 1


def test_local_relief_is_zero_on_a_plane_and_positive_on_a_mound():
    z = _plane(0.05, 0.02, 41)
    assert np.abs(accel.local_relief(z, 5)[8:-8, 8:-8]).max() < 1e-3
    rows, cols = np.mgrid[0:41, 0:41].astype(np.float32)
    mound = z + np.exp(-((rows - 20) ** 2 + (cols - 20) ** 2) / 18.0).astype(np.float32)
    assert accel.local_relief(mound, 8)[20, 20] > 0.5


def test_openness_and_sky_view_on_flat_ground():
    opn, svf = accel.openness_svf(np.zeros((21, 21), np.float32), 1.0, 5, 16)
    assert opn[10, 10] == pytest.approx(90.0, abs=1e-4)
    assert svf[10, 10] == pytest.approx(1.0, abs=1e-6)
