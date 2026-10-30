"""expds.las_grid_fast: Rust (expds_fast) и NumPy дают побайтно одну и ту же сетку.

Синтетические LAS (форматы 3 и 6, классы 2/5/7/18, точки за краем сетки), все три режима,
один кусок и много кусков (стыки сумм). Реальные облака — если задан EXPDS_LAS_GLOB.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from expds import las_grid_fast as lg

rust = pytest.mark.skipif(lg._fast is None, reason="expds_fast не установлен")


def _las(path, pf, n=200_000, seed=0):
    import laspy

    rng = np.random.default_rng(seed)
    hdr = laspy.LasHeader(point_format=pf, version="1.4" if pf >= 6 else "1.2")
    hdr.scales = np.array([0.01, 0.01, 0.001])
    hdr.offsets = np.array([500_000.0, 6_100_000.0, 0.0])
    las = laspy.LasData(hdr)
    x = rng.uniform(0, 300, n)
    y = rng.uniform(0, 200, n)
    las.x = 500_000.0 + x
    las.y = 6_100_000.0 + y
    las.z = 150.0 + 0.02 * x + rng.normal(0, 0.3, n)
    las.classification = rng.choice(np.array([2, 2, 2, 5, 7, 18], np.uint8), n)
    las.write(str(path))
    return str(path)


def _grid(path, cell=0.25, shrink=10.0):
    import laspy

    with laspy.open(path) as f:
        mn, mx = f.header.mins, f.header.maxs
    left, top = float(np.floor(mn[0])) + shrink, float(np.ceil(mx[1])) - shrink  # часть точек вне сетки
    cols = int(np.ceil((mx[0] - left) / cell)) + 1 - int(2 * shrink / cell)
    rows = int(np.ceil((top - mn[1]) / cell)) + 1 - int(2 * shrink / cell)
    return left, top, cell, rows, cols


@pytest.fixture(scope="module", params=[3, 6])
def las_path(request, tmp_path_factory):
    return _las(tmp_path_factory.mktemp("lg") / f"pf{request.param}.las", request.param, seed=request.param)


def _ref(path, left, top, cell, rows, cols, mode, chunk):
    """Эталон — тот же код, что ground.rasterise (NumPy)."""
    return lg.rasterise(path, left, top, cell, rows, cols, mode, chunk=chunk, use_rust=False)


@rust
@pytest.mark.parametrize("mode", ["class", "bare", "lowest"])
@pytest.mark.parametrize("chunk", [lg.STREAM_POINTS, 7_777])
@pytest.mark.parametrize("raw", [True, False])
def test_rust_equals_numpy(las_path, mode, chunk, raw, monkeypatch):
    if not raw:
        monkeypatch.setattr(lg, "_raw_layout", lambda arr: None)
    g = _grid(las_path)
    a = lg.rasterise(las_path, *g, mode, chunk=chunk)
    b = _ref(las_path, *g, mode, chunk)
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    assert a.tobytes() == b.tobytes()
    assert np.isfinite(a).any() and np.isnan(a).any()


@rust
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_accumulators_equal(las_path, mode):
    g = _grid(las_path)
    a = lg.accumulate(las_path, *g, mode, chunk=7_777, use_rust=True)
    b = lg.accumulate(las_path, *g, mode, chunk=7_777, use_rust=False)
    for u, v in zip(a, b):
        assert u.dtype == v.dtype and u.tobytes() == v.tobytes()


@rust
def test_grid_mean_bitexact():
    rng = np.random.default_rng(0)
    cnt = rng.integers(0, 5, 100_003).astype(np.int32)
    zsum = rng.normal(150, 40, cnt.size) * cnt
    ref = np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan).astype(np.float32)
    assert lg._fast.grid_mean(zsum, cnt).tobytes() == ref.tobytes()


REAL = sorted(glob.glob(os.environ.get("EXPDS_LAS_GLOB", "")))[:3] if os.environ.get("EXPDS_LAS_GLOB") else []


@rust
@pytest.mark.skipif(not REAL, reason="EXPDS_LAS_GLOB не задан")
@pytest.mark.parametrize("path", REAL)
@pytest.mark.parametrize("mode", ["class", "bare", "lowest"])
def test_real_clouds(path, mode):
    g = _grid(path, shrink=0.0)
    assert lg.rasterise(path, *g, mode).tobytes() == _ref(path, *g, mode, lg.STREAM_POINTS).tobytes()
