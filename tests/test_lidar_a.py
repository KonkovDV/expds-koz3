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
