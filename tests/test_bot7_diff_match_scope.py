"""build_curves_with_pairs + diff_match_scope — cross-layer TP видны в dump."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from expds import metric

ROOT = Path(__file__).resolve().parents[1]
METRIC = os.environ.get("EXPDS_METRIC") or str(ROOT / "materials" / "compute_metrics.py")
pytestmark = pytest.mark.skipif(not Path(METRIC).exists(), reason="нет compute_metrics.py")
X0, Y0 = 3_545_000.0, 7_310_000.0
K = "kurgany_tselye"


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setattr(metric, "_SHM_CACHE_DIR", tmp_path / "shm")
    monkeypatch.delenv("EXPDS_MATCH_SCOPE", raising=False)
    metric._UTM_GEOM_CACHE.clear()
    metric._SCOPE_WARNED.clear()
    yield


def sq(x, y, s=10.0):
    return {"type": "Polygon", "coordinates": [[[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]]}


def feat(geom, cls, layer=None, conf=None, region="R"):
    p = {"class_name": cls, "region_name": region}
    if layer is not None:
        p["markup_type"] = layer
    if conf is not None:
        p["confidence"] = conf
    return {"type": "Feature", "geometry": geom, "properties": p}


def fc(fs):
    return {"type": "FeatureCollection", "features": fs}


def test_with_pairs_marks_cross_layer():
    gt = fc([feat(sq(X0, Y0), K, "li")])
    pred = fc([feat(sq(X0, Y0), K, "ae", 0.9), feat(sq(X0 + 1, Y0 + 1), K, "li", 0.5)])
    g_curves, g_pairs = metric.build_curves_with_pairs(pred, gt, METRIC, scope="global")
    assert len(g_pairs) == 1
    assert g_pairs[0]["cross_layer"] is True
    assert g_pairs[0]["pred_layer"] == "ae" and g_pairs[0]["gt_layer"] == "li"
    assert g_curves[K].hit.tolist() == [True, False]

    metric._UTM_GEOM_CACHE.clear()
    l_curves, l_pairs = metric.build_curves_with_pairs(pred, gt, METRIC, scope="layer")
    assert len(l_pairs) == 1
    assert l_pairs[0]["cross_layer"] is False
    assert l_pairs[0]["pred_layer"] == "li"
    assert l_curves[K].hit.tolist() == [False, True]


def test_with_pairs_same_layer_not_cross():
    gt = fc([feat(sq(X0, Y0), K, "li")])
    pred = fc([feat(sq(X0, Y0), K, "la", 0.8)])  # alias → li
    _, pairs = metric.build_curves_with_pairs(pred, gt, METRIC, scope="global")
    assert len(pairs) == 1 and pairs[0]["cross_layer"] is False
    assert pairs[0]["pred_layer"] == "li" and pairs[0]["gt_layer"] == "li"


def test_diff_match_scope_script(tmp_path):
    gt = fc([feat(sq(X0, Y0), K, "li"), feat(sq(X0 + 80, Y0), K, "ae")])
    pred = fc([
        feat(sq(X0, Y0), K, "ae", 0.95),          # cross → steals li GT in global
        feat(sq(X0 + 80, Y0), K, "ae", 0.5),
        feat(sq(X0 + 1, Y0 + 1), K, "li", 0.4),
    ])
    pred_p = tmp_path / "pred.geojson"
    gt_p = tmp_path / "gt.geojson"
    out = tmp_path / "out"
    pred_p.write_text(json.dumps(pred), encoding="utf-8")
    gt_p.write_text(json.dumps(gt), encoding="utf-8")
    cmd = [sys.executable, str(ROOT / "scripts" / "diff_match_scope.py"),
           "--pred", str(pred_p), "--gt", str(gt_p), "--metric", METRIC, "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr + r.stdout
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_cross_layer_tp_global"] >= 1
    assert summary["delta"] <= 0
    assert (out / "cross_matches.csv").exists()
    assert (out / "delta_by_class.csv").exists()
    rows = (out / "cross_matches.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) >= 2  # header + ≥1
    assert "ae" in rows[1] and "li" in rows[1]


def test_reference_scope_default_and_layer_no_nameerror():
    """Регрессия d1a98ae: build_curves_reference не NameError(per_site); default==global; layer фильтрует."""
    gt = fc([feat(sq(X0, Y0), K, "li")])
    pred = fc([feat(sq(X0, Y0), K, "ae", 0.9)])
    # default (None) и явный global — без фильтра слоёв, hit=True
    d = metric.build_curves_reference(pred, gt, METRIC)  # scope default
    metric._UTM_GEOM_CACHE.clear()
    g = metric.build_curves_reference(pred, gt, METRIC, scope="global")
    assert d[K].hit.tolist() == g[K].hit.tolist() == [True]
    metric._UTM_GEOM_CACHE.clear()
    # layer: ae≠li → hit=False (фильтр keyed по layer_key)
    b = metric.build_curves_reference(pred, gt, METRIC, scope="layer")
    assert b[K].hit.tolist() == [False]
    metric._UTM_GEOM_CACHE.clear()
    # site: один участок R — как global на этом примере
    s = metric.build_curves_reference(pred, gt, METRIC, scope="site")
    assert s[K].hit.tolist() == [True]
