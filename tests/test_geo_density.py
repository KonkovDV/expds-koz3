"""Отсечка пиков находок с поправкой на густое поле (geophys.density_floor, peak_density_min)."""
import numpy as np
from conftest import UTM, X0, Y0
from rasterio.transform import from_origin

from expds import geophys

RES = 0.1
STRONG = [(2.0, 2.0), (8.0, 8.0)]
WEAK = [(2.0, 8.0), (5.0, 2.0), (5.0, 5.0), (8.0, 2.0), (5.0, 8.0), (2.0, 5.0)]
WEAK_AMP = [0.9, 0.85, 0.8, 0.7, 0.65, 0.6]


def bumps(size_m: float = 10.0) -> np.ndarray:
    """|hp| в MAD: две сильные аномалии и шесть слабых (ниже отсечки 1,0)."""
    n = int(size_m / RES)
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) * RES
    hp = np.zeros((n, n), np.float32)
    for (cx, cy), amp in [(c, 5.0) for c in STRONG] + list(zip(WEAK, WEAK_AMP)):
        hp += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.3 ** 2))
    return hp


def peaks(p: geophys.GeoParams) -> list[dict]:
    return geophys.peaks_from(bumps(), from_origin(X0, Y0 + 10.0, RES, RES), UTM, RES, "r", "maggr", "m.tif", p)


def test_density_floor_keeps_floor_when_enough_peaks():
    vals = np.array([5, 3, 0.9, 0.8, 0.7, 0.6], np.float32)
    assert geophys.density_floor(vals, 1.0, 0.5, 0) == 1.0
    assert geophys.density_floor(vals, 1.0, 0.5, 2) == 1.0
    assert np.isclose(geophys.density_floor(vals, 1.0, 0.5, 4), 0.8)
    assert geophys.density_floor(vals, 1.0, 0.5, 10) == 0.5


def test_off_by_default_and_lowers_only_sparse_maps():
    p = geophys.GeoParams(peak_radius_m=1.0, peak_k=1.0)
    assert p.peak_density_min == 0.0
    assert len(peaks(p)) == 2
    p.peak_density_min = 20.0
    assert len(peaks(p)) == 2
    p.peak_density_min = 50.0
    found = peaks(p)
    assert len(found) == 5
    assert min(f["confidence"] for f in found) > 0
    p.peak_density_min = 1000.0
    assert len(peaks(p)) == 8
    p.peak_k_min = 0.75
    assert len(peaks(p)) == 5
