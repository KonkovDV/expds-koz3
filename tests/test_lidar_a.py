"""expds.lidar_a: own invariants of the speedups (no reference copy needed)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _dtm(seed=0, shape=(203, 157)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    d = (100 + 0.05 * xx + 0.02 * yy + 1.5 * np.exp(-((xx - 70) ** 2 + (yy - 90) ** 2) / 300)).astype(np.float32)
    d += rng.normal(0, 0.05, shape).astype(np.float32)
    v = np.ones(shape, bool); v[:20, :30] = False; v[150:, 100:] = False
    return d, v


@pytest.mark.parametrize("window", [None, 64])
def test_base_stack_is_slice_of_extra(window):
    from expds.lidar_a import runner, terrain
    d, v = _dtm(1, (330, 290))
    kw = {} if window is None else dict(window=window, margin=24)
    x9, v9 = terrain.relief_stack_windowed(d, v, 0.25, extra=False, dev="cpu", **kw)
    x12, v12 = terrain.relief_stack_windowed(d, v, 0.25, extra=True, dev="cpu", **kw)
    assert x12[terrain.BASE_OF_EXTRA].tobytes() == x9.tobytes() and (v9 == v12).all()
    if window is not None:
        return
    got, gv = runner.stacks_for(d, v, 0.25, ["", "x"], "cpu")
    assert got[""].tobytes() == x9.tobytes() and got["x"].tobytes() == x12.tobytes() and (gv == v9).all()


def _las_set(tmp_path):
    from conftest import write_las
    return [write_las(tmp_path / f"s{i}.las", n=n, seed=i, ground_class=g) for i, (n, g) in enumerate([(40_000, 2), (90_000, 1), (20_000, 2)])]


@pytest.mark.parametrize("ordered", [True, False])
def test_dtm_pool_equals_sequential(tmp_path, ordered):
    from expds.lidar_a import ground
    from expds.lidar_a.dtm_workers import DtmPool
    paths = _las_set(tmp_path)
    seen = []
    for i, p, (d, v, o), st in DtmPool(paths, workers=2, ram_budget=300 << 20, ordered=ordered, tmpdir=str(tmp_path)):
        rd, rv, ro = ground.make_dtm(paths[i])
        assert d.tobytes() == rd.tobytes() and v.tobytes() == rv.tobytes() and o == ro
        seen.append(i)
    assert sorted(seen) == [0, 1, 2] and (seen == [0, 1, 2] or not ordered)
    assert not list(tmp_path.glob("*.npy"))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_all_equals_sequential(tmp_path):
    from expds.lidar_a import runner
    paths = _las_set(tmp_path)
    net = torch.nn.Conv2d(9, 2, 3, padding=1).eval()
    models = [(net, "", 1.0)]

    def decode(prob, vmask, meta):
        return [{"region": meta["region"], "sum": float(prob.astype(np.float64).sum()), "x0": meta["x0"]}]
    seq = {r: runner.predict(ps, models, 2, decode, meta={"region": r}, dev="cpu") for r, ps in (("A", paths[:2]), ("B", paths[2:]))}
    par = runner.predict_all([({"region": "A"}, paths[:2]), ({"region": "B"}, paths[2:])], models, 2, decode, dev="cpu", ram_budget=300 << 20)
    assert par == seq


# ---- (3) hole fill ---------------------------------------------------------------------------------------------------
def _ground(h=700, w=900, seed=0):
    """Sparse ground returns with gaps, a hole and a large empty strip (cells far from any ground)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (0.01 * xx + np.sin(yy / 23.0)).astype(np.float32)
    g[rng.random((h, w)) < 0.35] = np.nan
    g[200:260, 100:180] = np.nan
    g[:, 600:] = np.nan                     # 300 cells = 75 m of no ground
    return g


def test_d2_max():
    from expds.lidar_a import ground
    for res in (0.25, 0.5, 1.0, 0.3):
        t = ground._d2_max(res)
        d2 = np.arange(0, 4 * t + 10)
        assert np.array_equal(d2 <= t, np.sqrt(d2.astype(np.float64)) * res <= ground.FILL_LIMIT_M)


@pytest.mark.parametrize("rows", [37, 10 ** 6])
def test_fill_exact_fast_bitwise(rows, monkeypatch):
    from expds.lidar_a import ground
    g = _ground()
    ref, vref = ground.fill_holes(g, 0.25, "exact")
    monkeypatch.setattr(ground, "ROW_BLOCK", rows)
    out, v = ground.fill_holes(g, 0.25, "exact_fast")
    assert np.array_equal(v, vref) and out.tobytes() == ref.tobytes()


@pytest.mark.parametrize("tile,band_m", [(64, 6.0), (128, 20.0), (4096, 20.0)])
def test_fill_band_exact_within_band(tile, band_m):
    from scipy import ndimage
    from expds.lidar_a import ground
    g = _ground()
    has = np.isfinite(g)
    dist, (ry, rx) = ndimage.distance_transform_edt(~has, return_indices=True)
    iy_, ix_, v = ground._band_indices(has, 0.25, band_m=band_m, tile=tile, threads=2)
    assert np.array_equal(v, dist * 0.25 <= ground.FILL_LIMIT_M)
    near = dist * 0.25 <= band_m - 0.5      # ties at equal distance may pick another ground cell: compare values
    assert np.array_equal(g[iy_, ix_][near], g[ry, rx][near])
    assert has[iy_, ix_].all()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_band_fill_keeps_network_input(monkeypatch):
    """Band 80 m ⊇ receptive field of the relief stack: equal in value to the exact-fill stack (only the sign of
    value·0 on invalid pixels may differ), and so is a network's output."""
    from expds.lidar_a import ground, terrain, tiling
    g = _ground(700, 1400)
    d0, v0 = ground.fill_holes(g, 0.25, "exact")
    monkeypatch.setattr(ground, "TILE_PX", 256)
    d1, v1 = ground.fill_holes(g, 0.25, "band")
    assert np.array_equal(v0, v1) and not np.array_equal(d0, d1)
    x0, _ = terrain.relief_stack_windowed(d0, v0, 0.25, extra=True, dev="cpu")
    x1, _ = terrain.relief_stack_windowed(d1, v1, 0.25, extra=True, dev="cpu")
    assert np.array_equal(x0, x1)
    diff = x0.view(np.uint32) != x1.view(np.uint32)
    assert (x0[diff] == 0).all()
    net = torch.nn.Conv2d(12, 3, 5, padding=2).eval()
    p0 = tiling.infer_map(net, x0, 3, tile=256, overlap=64, tta=False, dev="cpu")
    p1 = tiling.infer_map(net, x1, 3, tile=256, overlap=64, tta=False, dev="cpu")
    assert np.array_equal(p0, p1)


# ---- (5) blending on the model's device ------------------------------------------------------------------------------
class _Net(torch.nn.Module):
    def __init__(self, cin, cout, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.nn.Conv2d(cin, 6, 3, padding=1); self.b = torch.nn.Conv2d(6, cout, 3, padding=1)
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.3

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("tta,shape", [(True, (9, 450, 350)), (False, (9, 300, 700)), ("d8", (9, 512, 512)), (True, (9, 100, 90))])
def test_device_blend_bitwise(tta, shape):
    from expds.lidar_a import tiling
    x = np.random.default_rng(3).normal(size=shape).astype(np.float32)
    net = _Net(9, 3).eval()
    ref = tiling.infer_map(net, x, 3, tile=256, overlap=64, tta=tta, dev="cpu")
    got = tiling.infer_map_on_device(net, x, 3, tile=256, overlap=64, tta=tta, dev="cpu").numpy()
    assert got.tobytes() == ref.tobytes()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_site_maps_accumulation_modes():
    from expds.lidar_a import runner
    d, v = _dtm(5, (300, 260))
    models = [(_Net(9, 3, 1).eval(), "", 0.5), (_Net(9, 3, 2).eval(), "", 0.5), (_Net(12, 3, 3).eval(), "x", 1.0)]
    host, vh = runner.site_maps(d, v, 0.25, models, 3, tta=True, dev="cpu", accumulate="host")
    exact, ve = runner.site_maps(d, v, 0.25, models, 3, tta=True, dev="cpu", accumulate="device_exact")
    dev_, vd = runner.site_maps(d, v, 0.25, models, 3, tta=True, dev="cpu", accumulate="device")
    assert exact.tobytes() == host.tobytes() and (vh == ve).all() and (vh == vd).all()
    assert np.abs(dev_ - host).max() <= 4 * np.spacing(np.float32(1))
