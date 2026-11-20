"""область сопоставления build_curves scope=layer — эталон только своего слоя
(markup_type / layer_key). По умолчанию global остаётся байт-идентичным старому поведению.

Слойные синонимы: la→li, german_ae→ae, spor→sp_or. maggr/etgr/terfl/geogr — отдельные.
Пустой markup_type → "?". Счётчики TP/FP/FN по слоям складываются (микро-F1).
"""
from __future__ import annotations

import os
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


def counts_from_curves(curves, tau=0.0):
    out = {}
    for c, cv in curves.items():
        k = cv.conf >= tau
        tp = int(cv.hit[k].sum())
        out[c] = (tp, int(k.sum()) - tp, cv.n_gt - tp)
    return out


def test_layer_key_aliases_and_unknown():
    assert metric.layer_key({"markup_type": "la"}) == "li"
    assert metric.layer_key({"markup_type": "LI"}) == "li"
    assert metric.layer_key({"markup_type": "german_ae"}) == "ae"
    assert metric.layer_key({"markup_type": "spor"}) == "sp_or"
    assert metric.layer_key({"markup_type": "sp_or"}) == "sp_or"
    assert metric.layer_key({"markup_type": "maggr"}) == "maggr"
    assert metric.layer_key({"markup_type": "etgr"}) == "etgr"
    assert metric.layer_key({"markup_type": "terfl"}) == "terfl"
    assert metric.layer_key({}) == "?"
    assert metric.layer_key({"markup_type": ""}) == "?"
    assert metric.layer_key({"markup_type": "  "}) == "?"
    assert metric.layer_key({"markup_type": "custom_x"}) == "custom_x"


def test_match_scope_accepts_layer(monkeypatch, capsys):
    assert metric.match_scope() == "global"
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "layer")
    assert metric.match_scope() == "layer"
    err = capsys.readouterr().err
    assert "scope=layer" in err
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "bogus")
    with pytest.raises(ValueError):
        metric.match_scope()


def test_cross_layer_blocked():
    """Чужой слой не забирает эталон: global даёт TP, layer — FP+FN."""
    gt = fc([feat(sq(X0, Y0), K, "li")])
    # более уверенный pred на другом слое (ae) + свой на li
    pred = fc([
        feat(sq(X0, Y0), K, "ae", 0.9),
        feat(sq(X0 + 1, Y0 + 1), K, "li", 0.5),
    ])
    g = metric.build_curves(pred, gt, METRIC, scope="global")[K]
    metric._UTM_GEOM_CACHE.clear()
    s = metric.build_curves(pred, gt, METRIC, scope="layer")[K]
    # global: жадно берёт ae (0.9) → hit=[True, False]
    assert g.hit.tolist() == [True, False]
    # layer: ae≠li → ae FP; li матчит свой эталон → hit=[False, True]
    assert s.hit.tolist() == [False, True]
    assert counts_from_curves({"K": g})["K"] == (1, 1, 0)
    assert counts_from_curves({"K": s})["K"] == (1, 1, 0)


def test_cross_layer_only_foreign():
    """Единственный pred на чужом слое: global TP, layer FP+FN."""
    gt = fc([feat(sq(X0, Y0), K, "li"), feat(sq(X0 + 50, Y0), K, "ae")])
    pred = fc([feat(sq(X0, Y0), K, "ae", 0.9)])
    g = counts_from_curves(metric.build_curves(pred, gt, METRIC, scope="global"))
    metric._UTM_GEOM_CACHE.clear()
    s = counts_from_curves(metric.build_curves(pred, gt, METRIC, scope="layer"))
    assert g[K] == (1, 0, 1)   # забрал чужой li-эталон
    assert s[K] == (0, 1, 2)   # ae-pred не матчит li; ae-gt без пары


def test_alias_la_matches_li():
    gt = fc([feat(sq(X0, Y0), K, "li")])
    pred = fc([feat(sq(X0, Y0), K, "la", 0.8)])
    hit = metric.build_curves(pred, gt, METRIC, scope="layer")[K].hit.tolist()
    assert hit == [True]


def test_missing_markup_only_matches_missing():
    gt = fc([feat(sq(X0, Y0), K, None), feat(sq(X0 + 40, Y0), K, "li")])
    pred = fc([feat(sq(X0, Y0), K, None, 0.9), feat(sq(X0 + 40, Y0), K, None, 0.8)])
    s = metric.build_curves(pred, gt, METRIC, scope="layer")[K]
    # оба pred с "?"; первый забирает gt "?"; второй не может взять li
    assert s.hit.tolist() == [True, False]
    assert counts_from_curves({"K": s})["K"] == (1, 1, 1)


def test_layer_micro_sum_two_layers():
    """Микро: TP/FP/FN складываются по слоям (как site по участкам)."""
    gt = fc([
        feat(sq(X0, Y0), K, "li"),
        feat(sq(X0 + 100, Y0), K, "ae"),
    ])
    pred = fc([
        feat(sq(X0, Y0), K, "li", 0.9),
        feat(sq(X0 + 100, Y0), K, "ae", 0.8),
        feat(sq(X0 + 200, Y0), K, "sp_or", 0.7),  # FP своего слоя
    ])
    c = counts_from_curves(metric.build_curves(pred, gt, METRIC, scope="layer"))
    assert c[K] == (2, 1, 0)


def test_default_global_unchanged_vs_explicit(monkeypatch):
    """Без флага (=global) hit-маска совпадает с явным scope=global."""
    gt = fc([feat(sq(X0, Y0), K, "li")])
    pred = fc([feat(sq(X0, Y0), K, "ae", 0.9), feat(sq(X0, Y0), K, "li", 0.4)])
    monkeypatch.delenv("EXPDS_MATCH_SCOPE", raising=False)
    a = metric.build_curves(pred, gt, METRIC)[K]
    metric._UTM_GEOM_CACHE.clear()
    b = metric.build_curves(pred, gt, METRIC, scope="global")[K]
    assert a.hit.tobytes() == b.hit.tobytes()
    assert a.conf.tobytes() == b.conf.tobytes()
    assert a.n_gt == b.n_gt


def test_env_switch_layer(monkeypatch):
    gt = fc([feat(sq(X0, Y0), K, "li")])
    pred = fc([feat(sq(X0, Y0), K, "ae", 0.9)])
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "layer")
    assert metric.build_curves(pred, gt, METRIC)[K].hit.tolist() == [False]
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "global")
    metric._UTM_GEOM_CACHE.clear()
    assert metric.build_curves(pred, gt, METRIC)[K].hit.tolist() == [True]
