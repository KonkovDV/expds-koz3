"""expds.lidar_a vs the reference implementation, bit for bit (skipped without the local reference copy)."""
from __future__ import annotations

import numpy as np
import pytest

from lidar_a_oracle import oracle, oracle_relief

torch = pytest.importorskip("torch")


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def _bare_las(path, n=60_000, seed=3):
    import laspy
    from conftest import X0, Y0, mound_surface
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 60, n), rng.uniform(0, 50, n)
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001]); hdr.offsets = np.array([X0, Y0, 0.0])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = x + X0, y + Y0, mound_surface(x, y)
    las.classification = np.full(n, 8, np.uint8)
    las.write(str(path))
    return path


@pytest.mark.parametrize("fill", ["exact", "exact_fast"])
@pytest.mark.parametrize("case", ["class", "lowest", "bare"])
def test_dtm_bitwise(tmp_path, case, fill, monkeypatch):
    from conftest import write_las
    from expds.lidar_a import ground
    monkeypatch.setattr(ground, "FILL_MODE", fill)
    ref = oracle("dtm")
    if case == "bare":
        p = _bare_las(tmp_path / "b.las")
    else:
        p = write_las(tmp_path / "a.las", n=60_000, ground_class=2 if case == "class" else 1)
    assert ground.ground_mode(p) == case
    d, v, o = ground.make_dtm(p)
    rd, rv, ro = ref.build(str(p))
    _same(d, rd); _same(v, rv); assert o == ro


def _dtm(seed=0, shape=(203, 157)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    d = (100 + 0.05 * xx + 0.02 * yy + 1.5 * np.exp(-((xx - 70) ** 2 + (yy - 90) ** 2) / 300)).astype(np.float32)
    d += rng.normal(0, 0.05, shape).astype(np.float32)
    v = np.ones(shape, bool); v[:20, :30] = False; v[150:, 100:] = False
    return d, v


@pytest.mark.parametrize("cell", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("kind", ["9", "12", "12u"])
def test_relief_bitwise(cell, kind):
    from expds.lidar_a import terrain
    rel = oracle_relief(oracle("relief"), extra=kind != "9", unclip=kind == "12u")
    d, v = _dtm(shape=(203, 157) if cell < 1 else (61, 47))
    got = terrain.relief_stack(d, v, cell, extra=kind != "9", unclip=kind == "12u", dev="cpu")
    exp = rel.channels(d, v, cell, dev="cpu")
    _same(got[0], exp[0]); _same(got[1], exp[1])


@pytest.mark.parametrize("extra", [False, True])
def test_relief_windowed_bitwise(extra):
    from expds.lidar_a import terrain
    rel = oracle_relief(oracle("relief"), extra=extra)
    d, v = _dtm(1, (330, 290))
    got = terrain.relief_stack_windowed(d, v, 0.25, extra=extra, dev="cpu", window=64, margin=24)
    exp = rel.channels_tiled(d, v, 0.25, tile_px=64, margin_px=24, dev="cpu")
    _same(got[0], exp[0]); _same(got[1], exp[1])


class _Net(torch.nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.a = torch.nn.Conv2d(cin, 8, 3, padding=1); self.b = torch.nn.Conv2d(8, cout, 3, padding=1)
        for p in self.parameters():
            p.data = torch.randn(p.shape, generator=g) * 0.3

    def forward(self, x):
        return self.b(torch.relu(self.a(torch.nn.functional.avg_pool2d(x, 2).repeat_interleave(2, -1).repeat_interleave(2, -2) + x)))


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("tta", [False, True, "d8"])
@pytest.mark.parametrize("hw,tile,ov", [((150, 97), 64, 16), ((64, 64), 64, 16), ((40, 300), 96, 24)])
def test_tiled_inference_bitwise(tta, hw, tile, ov):
    from expds.lidar_a import tiling
    hm = oracle("hm")
    x = np.random.default_rng(2).normal(size=(3,) + hw).astype(np.float32)
    net = _Net(3, 2).eval()
    with torch.no_grad():
        exp = hm.predict_tiled(net, x, 2, tile=tile, overlap=ov, tta=tta, dev="cpu")
    got = tiling.infer_map(net, x, 2, tile=tile, overlap=ov, tta=tta, dev="cpu")
    _same(got, exp)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_site_maps_vs_reference_pipeline():
    """Ensemble of a '' and an 'x' model = reference relief + reference tiling + float32-normalised tensordot."""
    from expds.lidar_a import runner
    hm, rel = oracle("hm"), oracle("relief")
    d, v = _dtm(4, (180, 140))
    models = [(_Net(9, 2).eval(), "", 0.5), (_Net(12, 2).eval(), "x", 1.0)]
    got, gv = runner.site_maps(d, v, 0.25, models, 2, tta=True, dev="cpu")
    with torch.no_grad():
        x9, vv = oracle_relief(rel, False).channels_tiled(d, v, 0.25, dev="cpu")
        x12 = oracle_relief(rel, True).channels_tiled(d, v, 0.25, dev="cpu")[0]
        oracle_relief(rel, False)
        ps = [hm.predict_tiled(m, xx, 2, tile=1024, overlap=192, tta=True, dev="cpu") for (m, _, _), xx in zip(models, (x9, x12))]
    ws = np.array([0.5, 1.0], np.float32)
    _same(got, np.tensordot(ws / ws.sum(), np.stack(ps), axes=1)); _same(gv, vv)
