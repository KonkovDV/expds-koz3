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
