import numpy as np
import pytest

from conftest import MOUNDS, UTM, X0, Y0, write_las
from expds import accel, candidates
from expds.lidar import build_dem, hillshade_set


def test_las_grid_backends_agree(tmp_path):
    las = write_las(tmp_path / "a.las")
    a, ia = accel.las_grid(las, 0.25)
    with_np = accel._rs
    accel._rs = None
    try:
        b, ib = accel.las_grid(las, 0.25)
    finally:
        accel._rs = with_np
    assert a.shape == b.shape
    assert np.allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), atol=1e-4)
    assert ia["points_used"] == ib["points_used"] and ia["left"] == pytest.approx(ib["left"])


def test_ground_fallback_to_all_points(tmp_path):
    las = write_las(tmp_path / "b.las", ground_class=1)   # нет класса 2
    dem, info = accel.las_grid(las, 0.5)
    assert info["used_all"] and np.isfinite(dem).any()


def test_dem_and_hillshades(tmp_path):
    las = write_las(tmp_path / "c.las")
    d = build_dem(las, res=0.1)
    assert d.crs == UTM
    assert d.info["holes"] > 0 and np.isfinite(d.dem).mean() > 0.95
    imgs = hillshade_set(d)
    assert set(imgs) == {"c", "ch", "g", "i"}
    assert imgs["g"].dtype == np.uint8 and imgs["c"].shape[0] == 3
    # i = 255 − g внутри съёмки
    v = np.isfinite(d.dem)
    assert np.array_equal(imgs["i"][v].astype(int), 255 - imgs["g"][v].astype(int))


def test_candidates_find_mounds(tmp_path):
    las = write_las(tmp_path / "d.las", n=200_000)
    d = build_dem(las, res=0.1)
    c = candidates.detect(d)
    assert len(c["x"]) >= len(MOUNDS)
    s = candidates.score(c["features"])
    top = np.argsort(-s)[: len(MOUNDS)]
    found = 0
    for mx, my, _, _ in MOUNDS:
        dist = np.hypot(c["x"][top] - (X0 + mx), c["y"][top] - (Y0 + my))
        found += dist.min() < 1.5
    assert found == len(MOUNDS)
