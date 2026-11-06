"""Свой модуль геофизики обязан совпасть с расчётом, на котором держится балл.

Сравниваются признаки и круги находок на синтетической карте. Деревья те же,
поэтому совпадение признаков сохраняет вероятности и официальный балл.
"""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
BLOB = (
    "archive/worked-20261031:archive_review/analysis/2026-10-29/bots/bot-4/"
    "runs/review/v2eb268d6/expds/geo_pixgbm.py"
)


def _reference(tmp_path):
    got = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", BLOB],
        capture_output=True,
    )
    if got.returncode != 0:
        pytest.skip("эталонный расчёт не в этом клоне")
    raw = subprocess.check_output(["git", "-C", str(ROOT), "cat-file", "-p", BLOB])
    path = tmp_path / "ref_pix.py"
    path.write_bytes(raw)
    spec = importlib.util.spec_from_file_location("expds.ref_pix_score", path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["expds.ref_pix_score"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_features_and_finds_match_the_scoring_map(tmp_path):
    ref = _reference(tmp_path)
    from expds import pix_own as own

    rng = np.random.default_rng(0)
    h, w = 48, 40
    valid = np.ones((h, w), bool)
    valid[:2, :] = False
    z = rng.normal(size=(h, w)).astype(np.float32)
    zd = rng.normal(size=(h, w)).astype(np.float32)
    (c5, c15), b = own.trends(z, valid, 0.1)
    y0, x0, y1, x1 = 0, 0, h, w
    rr = np.array([10, 20, 30], np.int64)
    cc = np.array([8, 12, 18], np.int64)
    kw = dict(gsd=0.1, mt_id=0, rr=rr, cc=cc)
    a = own.features_at(z, zd, valid, own._up(c5, b, y0, x0, h, w), own._up(c15, b, y0, x0, h, w), **kw)
    rb = ref.trends(z, valid, 0.1)
    b5, b15 = rb[0]
    bb = rb[1]
    c = ref.features_at(z, zd, valid, ref._up(b5, bb, y0, x0, h, w), ref._up(b15, bb, y0, x0, h, w), **kw)
    assert set(a) == set(c)
    for name in a:
        assert np.allclose(a[name], c[name], rtol=0, atol=0), name

    from rasterio.transform import Affine

    p = np.zeros((h, w), np.float32)
    p[12, 12] = 0.9
    p[12, 13] = 0.4
    p[30, 8] = 0.5
    d = {"gsd": 0.1, "T": Affine(0.1, 0, 1000.0, 0, -0.1, 2000.0), "crs": "EPSG:3857"}
    fa = own.finds_candidates(p, d, "r", "maggr", "a.tif")
    fb = ref.finds_candidates(p, d, "r", "maggr", "a.tif")
    assert len(fa) == len(fb) == 2
    for x, y in zip(fa, fb):
        assert x["confidence"] == y["confidence"]
        assert x["geom"].wkt == y["geom"].wkt
