"""Крутой склон отсекается, плоскость остаётся. Сданные каналы лидара не меняются."""
import numpy as np

from expds.lidar import ENCLOSURE_TAGS, RELIEF_TAGS, Dem
from expds.slope_gate import gentle_mask


def test_shipped_tags_are_not_the_enclosure_set():
    assert RELIEF_TAGS == ("lrm_s", "lrm_l", "slope")
    assert "rough" in ENCLOSURE_TAGS and "curv" in ENCLOSURE_TAGS
    assert RELIEF_TAGS != ENCLOSURE_TAGS


def test_steep_ramp_is_dropped_and_flat_ground_is_kept():
    n = 41
    res = 1.0
    ramp = (np.tan(np.deg2rad(20.0)) * np.arange(n)[None, :]).astype(np.float32)
    ramp = np.repeat(ramp, n, axis=0)
    flat = np.zeros((n, n), np.float32)
    left, top = 0.0, float(n)
    x = np.array([left + (n // 2 + 0.5) * res])
    y = np.array([top - (n // 2 + 0.5) * res])
    steep = Dem(dem=ramp, left=left, top=top, res=res, crs="EPSG:32636")
    plain = Dem(dem=flat, left=left, top=top, res=res, crs="EPSG:32636")
    assert gentle_mask(steep, x, y, 12.0, res).tolist() == [False]
    assert gentle_mask(plain, x, y, 12.0, res).tolist() == [True]
    assert gentle_mask(plain, np.array([]), np.array([]), 12.0, res).shape == (0,)
