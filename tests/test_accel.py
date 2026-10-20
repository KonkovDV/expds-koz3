"""Rust и NumPy дают одинаковый результат; NumPy совпадает с OpenCV там, где бейзлайн его использует."""
from contextlib import contextmanager

import numpy as np
import pytest

from expds import accel

rust = pytest.mark.skipif(not accel.HAVE_RUST, reason="expds_core не собран")


@contextmanager
def numpy_backend():
    saved = accel._rs
    accel._rs = None
    try:
        yield
    finally:
        accel._rs = saved


def both(fn, *args, **kw):
    a = fn(*args, **kw)
    with numpy_backend():
        b = fn(*args, **kw)
    return a, b


def dem_with_holes(h=97, w=131, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    z = (100 + 0.05 * xx + 0.02 * yy + 2 * np.exp(-((xx - 60) ** 2 + (yy - 40) ** 2) / 50)).astype(np.float32)
    z += rng.normal(0, 0.01, z.shape).astype(np.float32)
    z[rng.random(z.shape) < 0.2] = np.nan
    z[10:20, 30:50] = np.nan
    return z


@pytest.mark.skipif(not hasattr(accel._rs, "fill_tin"), reason="expds_core без fill_tin")
def test_fill_tin_matches_scipy_on_plane():
    from expds import lidar

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:90, 0:120]
    plane = (50 + 0.3 * xx - 0.2 * yy).astype(np.float32)
    dem = plane.copy()
    dem[rng.random(dem.shape) < 0.7] = np.nan
    dem[30:55, 40:80] = np.nan
    valid = np.isfinite(dem)
    to_fill = accel.convex_hull_mask(valid) & ~valid
    tin = accel.fill_tin(dem)
    with numpy_backend():
        ref = lidar._fill_by_delaunay(dem, valid, to_fill, 10**9)
    assert np.array_equal(np.isfinite(tin) & to_fill, np.isfinite(ref) & to_fill)
    m = to_fill & np.isfinite(tin)
    assert m.sum() > 1000
    assert np.abs(tin[m] - plane[m]).max() < 1e-3 and np.abs(ref[m] - plane[m]).max() < 1e-3


@rust
def test_fill_pyramid_parity():
    a, b = both(accel.fill_pyramid, dem_with_holes())
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.allclose(a, b, atol=1e-4)


def test_resize_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    src = np.random.default_rng(1).normal(size=(13, 17)).astype(np.float32)
    for (h, w) in [(26, 34), (25, 33), (100, 7)]:
        ref = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
        with numpy_backend():
            ours = accel.resize_linear(src, h, w)
        assert np.allclose(ours, ref, atol=1e-5)
        if accel.HAVE_RUST:
            assert np.allclose(accel.resize_linear(src, h, w), ref, atol=1e-5)


def test_convex_hull_close_to_opencv():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    valid = np.zeros((120, 160), bool)
    yy, xx = np.mgrid[0:120, 0:160]
    valid[((xx - 80) / 60) ** 2 + ((yy - 60) / 40) ** 2 < 1] = True
    valid &= rng.random(valid.shape) < 0.7
    rows, cols = np.nonzero(valid)
    pts = np.column_stack([cols, rows]).astype(np.int32).reshape(-1, 1, 2)
    ref = np.zeros(valid.shape, np.uint8)
    cv2.fillConvexPoly(ref, cv2.convexHull(pts), 1)
    for m in both(accel.convex_hull_mask, valid) if accel.HAVE_RUST else [accel.convex_hull_mask(valid)]:
        assert (m == ref.astype(bool)).mean() > 0.995


@rust
def test_hillshade_and_color_parity():
    z = accel.fill_pyramid(dem_with_holes())
    a, b = both(accel.hillshade, z, 0.1, 0.1)
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.max() <= 1 and (d > 0).mean() < 0.01
    valid = np.ones(z.shape, bool)
    valid[:3] = False
    zmin, zr = float(z.min()), float(z.max() - z.min())
    a, b = both(accel.color_relief, z, valid, zmin, zr, 3.0)
    d = np.abs(a.astype(int) - b.astype(int))
    assert a.shape == (3,) + z.shape and d.max() <= 1 and (d > 0).mean() < 0.01
    assert (a[:, :3] == 0).all()


@rust
def test_relief_peaks_ring_parity():
    z = accel.fill_pyramid(dem_with_holes())
    a, b = both(accel.local_relief, z, 7)
    assert np.allclose(a, b, atol=1e-4)
    (r1, c1, v1), (r2, c2, v2) = both(accel.find_peaks, a, 4, 0.05)
    assert len(r1) > 0 and np.array_equal(r1, r2) and np.array_equal(c1, c2)
    s1, s2 = both(accel.ring_stats, z, r1, c1, 2.0, 6.0)
    assert np.allclose(s1, s2, atol=1e-3, equal_nan=True)


def test_robust_z_matches_numpy():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(40, 70)).astype(np.float32)
    a[::5, ::3] = np.nan
    a[0, 0] = 1e6
    got, ref = both(accel.robust_z, a)
    assert np.allclose(got, ref, equal_nan=True, atol=1e-5)
    even = np.array([[1, 2], [3, 4]], np.float32)
    got, ref = both(accel.robust_z, even)
    assert np.allclose(got, ref, equal_nan=True, atol=1e-5)


def test_box_mean_ignores_nan():
    z = np.ones((5, 5), np.float32)
    z[2, 2] = np.nan
    m = accel.box_mean(z, 1)
    assert np.allclose(m, 1.0)


def test_sample_window_matches_scipy():
    scipy = pytest.importorskip("scipy")
    from scipy.ndimage import maximum_filter

    if not accel.HAVE_RUST or not hasattr(accel._rs, "sample_window_max"):
        pytest.skip("нет sample_window_max")
    rng = np.random.default_rng(0)
    img = rng.normal(size=(3, 9, 11)).astype(np.float32)
    img[0, 2, 3] = np.nan
    rows = np.array([0, 1, 8, 4, 3], np.int64)
    cols = np.array([0, 10, 2, 5, 1], np.int64)
    got = accel._rs.sample_window_max(np.ascontiguousarray(img), rows, cols, 5)
    ref = np.stack([maximum_filter(p, size=5)[rows, cols] for p in img], 1)
    assert np.allclose(got, ref, equal_nan=True)
    assert scipy is not None


def test_relief_channels_match_numpy():
    from expds.lidar import Dem, relief_channels

    rng = np.random.default_rng(1)
    dem = rng.normal(scale=0.2, size=(24, 31)).astype(np.float32) + 100
    dem[0, :] = np.nan
    d = Dem(dem=dem, left=100.0, top=200.0, res=0.1, crs="EPSG:32636")
    a, ta = relief_channels(d, 0.5)
    with numpy_backend():
        b, tb = relief_channels(d, 0.5)
    assert ta.a == pytest.approx(float(tb.a))
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def test_slope_matches_numpy():
    z = dem_with_holes(48, 63)
    a, b = both(accel.slope_deg, z, 0.5)
    assert a.shape == z.shape
    assert np.allclose(a, b, atol=1e-4, equal_nan=True)


def test_work_grid_matches_numpy():
    from expds import candidates

    z = dem_with_holes(40, 55)
    for src, dst in ((0.1, 0.5), (0.3, 0.5), (0.5, 0.5)):
        (a, sa), (b, sb) = both(candidates.to_work_grid, z, src, dst)
        assert sa == pytest.approx(sb)
        assert a.shape == b.shape
        assert np.allclose(a, b, equal_nan=True, atol=1e-5)
