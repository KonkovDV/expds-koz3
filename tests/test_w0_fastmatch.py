"""EXPDS_RUST_MATCH=1: Rust-префильтр пар даёт те же hit и пары (i, j, score), что полный перебор STRtree."""
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box

from expds import accel, fastmatch

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not hasattr(accel._rs, "match_prefilter"), reason="expds_core без match_prefilter")
_spec = importlib.util.spec_from_file_location("bench_w0_fastmatch", ROOT / "scripts" / "bench_w0_fastmatch.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def _scene(seed):
    rnd = random.Random(seed)
    gts, preds = [], []
    for _ in range(150):
        x, y = rnd.uniform(5e5, 5.01e5), rnd.uniform(6.2e6, 6.201e6)
        k = rnd.random()
        if k < 0.3:
            gts.append(Point(x, y))
        elif k < 0.4:
            gts.append(MultiPolygon([box(x, y, x + 8, y + 5), box(x + 20, y, x + 25, y + 6)]))
        else:
            gts.append(Point(x, y).buffer(rnd.uniform(2, 15), 6))
    for _ in range(1500):
        g = rnd.choice(gts)
        c = g.centroid
        x, y = c.x + rnd.gauss(0, 6), c.y + rnd.gauss(0, 6)
        r = rnd.uniform(1, 12)
        k = rnd.random()
        if k < 0.05:   # ровно пол-покрытия: полоса половины эталона-квадрата
            preds.append(box(c.x - 4, c.y - 4, c.x, c.y + 4))
        elif k < 0.1:
            preds.append(Polygon([(x, y), (x + r, y), (x, y + r), (x + r, y + r)]))   # невалидный «бантик»
        else:
            preds.append(Point(x, y).buffer(r, 8))
    preds += [box(c.centroid.x - 4, c.centroid.y - 4, c.centroid.x + 4, c.centroid.y + 4) for c in gts[:20]]
    return preds, gts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_equals_strtree(monkeypatch, seed):
    preds, gts = _scene(seed)
    monkeypatch.setenv("EXPDS_RUST_MATCH", "1")
    monkeypatch.setattr(fastmatch, "MIN_PAIRS", 0)
    q = fastmatch.prefilter(preds, gts, B.calc)
    assert q is not None
    ref_q = shapely.STRtree([g.buffer(B.calc.r_tolerance) if g.geom_type == "Point" else g for g in gts]).query(preds)
    h1, p1 = B.greedy(preds, gts, ref_q)
    h2, p2 = B.greedy(preds, gts, q)
    assert np.array_equal(h1, h2) and p1 == p2 and h1.sum() > 0
    assert len(q[0]) <= len(ref_q[0])


def test_flag_off(monkeypatch):
    monkeypatch.setenv("EXPDS_RUST_MATCH", "0")
    assert fastmatch.prefilter([box(0, 0, 1, 1)], [box(0, 0, 1, 1)], B.calc) is None
