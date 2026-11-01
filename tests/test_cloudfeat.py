"""expds.cloudfeat: no LAS → NaN, CLI format, synthetic LAS → finite, deterministic values."""
import csv

import numpy as np
import pytest

from expds import cloudfeat as cf


def _cands(p, rows):
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cid", "cx", "cy", "w", "h"])
        w.writerows(rows)


def test_no_las_nan(tmp_path):
    _cands(tmp_path / "c.csv", [["a", 4.0e6, 7.5e6, 10, 12], ["b", 4.0e6 + 50, 7.5e6, 3, 3]])
    out = tmp_path / "cf.csv"
    assert cf.main(["--site", str(tmp_path), "--cands", str(tmp_path / "c.csv"), "--out", str(out)]) == 0
    rows = list(csv.reader(open(out)))
    assert rows[0] == ["cid", *cf.NAMES]
    assert [r[0] for r in rows[1:]] == ["a", "b"] and all(v == "" for r in rows[1:] for v in r[1:])
    assert np.isnan(cf.compute([], "EPSG:32637", np.ones((3, 4)))).all()


def test_synthetic_las(tmp_path):
    laspy = pytest.importorskip("laspy")
    pyproj = pytest.importorskip("pyproj")
    rng = np.random.default_rng(0)
    x0, y0 = 400000.0, 6200000.0
    n = 20000
    x = x0 + rng.uniform(0, 60, n)
    y = y0 + rng.uniform(0, 60, n)
    z = 100 + rng.normal(0, 0.2, n)
    c = np.where(rng.uniform(size=n) < 0.6, 2, 1).astype(np.uint8)
    h = laspy.LasHeader(point_format=1, version="1.2")
    h.offsets, h.scales = [x0, y0, 0], [0.01, 0.01, 0.01]
    las = laspy.LasData(h)
    las.x, las.y, las.z, las.classification = x, y, z, c
    p = tmp_path / "s.las"
    las.write(str(p))
    tr = pyproj.Transformer.from_crs(32637, 3857, always_xy=True)
    cx, cy = tr.transform(x0 + 30, y0 + 30)
    B = np.array([[cx, cy, 10, 10], [cx + 1e6, cy, 10, 10]])
    F1 = cf.compute([p], "EPSG:32637", B)
    F2 = cf.compute([p], "EPSG:32637", B)
    assert F1.shape == (2, len(cf.NAMES)) and F1.dtype == np.float32
    assert np.isfinite(F1[0]).all() and F1[0, 0] > 0
    # box outside the cloud: densities 0, hole share 1, z-stats NaN (same as training)
    assert F1[1, 0] == 0 and F1[1, 7] == 1 and np.isnan(F1[1, 5])
    np.testing.assert_array_equal(F1, F2)
