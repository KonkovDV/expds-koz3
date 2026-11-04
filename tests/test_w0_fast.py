"""w0: быстрые пути метрики/JSON дают тот же результат, что прежние (флаги выключения)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import shapely

import importlib
import sys

_DEV = os.environ.get("EXPDS_CORE_DEV")  # каталог со свежесобранным expds_core (conftest ставит inference/lib первым)
if _DEV:
    sys.path.insert(0, _DEV)
    sys.modules.pop("expds_core", None)
from expds import fastgeom, fastjson, metric  # noqa: E402

if _DEV:
    fastgeom = importlib.reload(fastgeom)

ROOT = Path(__file__).resolve().parents[1]
METRIC = str(ROOT / "materials" / "compute_metrics.py")
REAL_PRED = Path(os.environ.get("W0_REAL_PRED", "/workspace/koz3/w2/oof/merged/oof_cnx_c3-sc_s0.geojson"))
REAL_GT = Path(os.environ.get("W0_REAL_GT", "/workspace/koz3/ds/gt/gt_train_v2.geojson"))


def _fc(n=600, seed=0):
    rng = np.random.default_rng(seed)
    feats = []
    classes = ["kurgany_tselye", "kurgany_povrezhdennye", "fortifikatsii", "gorodishcha"]
    for i in range(n):
        cx, cy = 3.6e6 + rng.uniform(0, 2e4), 7.6e6 + rng.uniform(0, 2e4)
        r = rng.uniform(3, 40)
        k = int(rng.integers(3, 14))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        ring = [[float(cx + r * np.cos(a)), float(cy + r * np.sin(a))] for a in ang]
        if i % 3:
            ring.append(ring[0])
        g = {"type": "Polygon", "coordinates": [ring]}
        if i % 17 == 0:
            g = {"type": "Point", "coordinates": [cx, cy]}
        if i % 29 == 0:
            g = {"type": "MultiPolygon", "coordinates": [[ring + [ring[0]]]]}
        if i % 41 == 0:  # самопересечение → официальная ветка
            g = {"type": "Polygon", "coordinates": [[[cx, cy], [cx + 10, cy + 10], [cx + 10, cy], [cx, cy + 10], [cx, cy]]]}
        feats.append({"type": "Feature", "geometry": g, "properties": {
            "class_name": classes[i % 4], "confidence": float(rng.uniform()), "region_name": f"r{i % 7}",
            "markup_type": "li", "crs": "EPSG:3857"}})
    return {"type": "FeatureCollection", "features": feats}


def _curves(pred, gt, flag, monkeypatch):
    monkeypatch.setenv("EXPDS_FASTGEOM", flag)
    monkeypatch.setattr(metric, "_todo_batch_hash", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    metric._UTM_GEOM_CACHE.clear()
    return metric.build_curves(pred, gt, METRIC)


def _eq(a, b):
    assert sorted(a) == sorted(b)
    for c in a:
        x, y = a[c], b[c]
        assert np.array_equal(x.conf, y.conf) and np.array_equal(x.hit, y.hit), c
        assert x.pred_region == y.pred_region and x.pred_group == y.pred_group, c
        assert x.n_gt == y.n_gt and x.gt_regions == y.gt_regions, c


@pytest.mark.skipif(not fastgeom.enabled(), reason="expds_core с flatten_polygons не собран")
def test_fastgeom_synthetic_curves_identical(monkeypatch):
    pred, gt = _fc(800, 1), _fc(300, 2)
    _eq(_curves(pred, gt, "0", monkeypatch), _curves(pred, gt, "1", monkeypatch))


@pytest.mark.skipif(not fastgeom.enabled(), reason="expds_core с flatten_polygons не собран")
def test_fastgeom_wkb_identical():
    mod = metric.load_official(METRIC)
    feats = _fc(500, 3)["features"]
    os.environ["EXPDS_FASTGEOM"] = "0"
    ref = [mod._extract_geometry(f) for f in feats]
    os.environ["EXPDS_FASTGEOM"] = "1"
    new = fastgeom.extract_many(mod, feats)
    assert [None if g is None else shapely.to_wkb(g) for g in ref] == \
           [None if g is None else shapely.to_wkb(g) for g in new]


@pytest.mark.skipif(not (REAL_PRED.exists() and REAL_GT.exists() and fastgeom.enabled()), reason="нет реальных данных")
def test_fastgeom_real_curves_identical(monkeypatch):
    pred, gt = fastjson.load_json(REAL_PRED), fastjson.load_json(REAL_GT)
    _eq(_curves(pred, gt, "0", monkeypatch), _curves(pred, gt, "1", monkeypatch))


def test_load_json_equal(tmp_path):
    fc = _fc(200, 4)
    p = tmp_path / "a.geojson"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(fc).encode())
    assert fastjson.load_json(p) == fc


def test_filter_conf_rust_equals_ijson(tmp_path, monkeypatch):
    pytest.importorskip("ijson")
    fc = _fc(3000, 5)
    fc["features"][5]["properties"].pop("confidence")
    fc["features"][6]["properties"]["score"] = 0.9
    fc["features"][6]["properties"].pop("confidence")
    src = tmp_path / "src.geojson"
    src.write_text(json.dumps(fc, ensure_ascii=False, indent=1))
    a, b = tmp_path / "rs.geojson", tmp_path / "py.geojson"
    r1 = fastjson.filter_conf(src, a, 0.3)
    monkeypatch.setenv("EXPDS_FASTJSON", "0")
    r2 = fastjson.filter_conf(src, b, 0.3)
    assert r1 == r2
    assert json.loads(a.read_text()) == json.loads(b.read_text())
