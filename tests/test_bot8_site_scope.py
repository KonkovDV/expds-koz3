"""область сопоставления build_curves — global (как calculate_f1 на всём файле) и
site (эталон только своего участка, как calculate_f1 по каждому участку отдельно).

Эталоны — официальный materials/compute_metrics.py без изменений:
  global  ↔ F1Calculator().calculate_f1(pred, gt)            (счётчики TP/FP/FN каждого класса);
  site    ↔ metric.score_by_site: calculate_f1 на каждом участке split_merged_by_source_and_class,
            счётчики сложены.
Проверка — точное равенство целых счётчиков и итогового балла (==, без допуска).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from expds import metric

ROOT = Path(__file__).resolve().parents[1]
METRIC = os.environ.get("EXPDS_METRIC") or str(ROOT / "materials" / "compute_metrics.py")
B8 = Path("/workspace/koz3_shared/bot-8")
CROSS_PRED, CROSS_GT = B8 / "data" / "stack_cross.geojson", B8 / "data" / "gt_cross.geojson"
pytestmark = pytest.mark.skipif(not Path(METRIC).exists(), reason="нет compute_metrics.py")

X0, Y0 = 3_545_000.0, 7_310_000.0   # EPSG:3857, район Гнёздова


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setattr(metric, "_SHM_CACHE_DIR", tmp_path / "shm")
    monkeypatch.delenv("EXPDS_MATCH_SCOPE", raising=False)
    metric._UTM_GEOM_CACHE.clear()
    yield


def sq(x, y, s=10.0):
    return {"type": "Polygon", "coordinates": [[[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]]}


def feat(geom, cls, region=None, conf=None, sub=None):
    p = {"class_name": cls}
    if region is not None:
        p["region_name"] = region
    if sub is not None:
        p["sub_region_name"] = sub
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


def counts_official(res):
    return {c: (int(v["tp"]), int(v["fp"]), int(v["fn"])) for c, v in res["per_class"].items()}


def filt(pred, tau):
    mod = metric.load_official(METRIC)
    return fc([f for f in pred["features"] if mod._get_confidence(f) >= tau])


def check_both(pred, gt, taus=(0.0,)):
    """Возвращает счётчики (global, site) и сверяет каждый режим с официальным эталоном."""
    res = {}
    for scope in ("global", "site"):
        metric._UTM_GEOM_CACHE.clear()
        curves = metric.build_curves(pred, gt, METRIC, scope=scope)
        for tau in taus:
            off = (metric.score(filt(pred, tau), gt, METRIC) if scope == "global"
                   else metric.score_by_site(filt(pred, tau), gt, METRIC))
            mine = counts_from_curves(curves, tau)
            assert mine == counts_official(off), (scope, tau)
            fast, _ = metric.score_at(curves, {c: tau for c in curves})
            assert fast == off["final_score"] or abs(fast - off["final_score"]) < 1e-15, (scope, tau)
        res[scope] = counts_from_curves(curves)
    return res


K = "kurgany_tselye"


def test_cross_site_polygon():
    gt = fc([feat(sq(X0, Y0), K, "B")])
    pred = fc([feat(sq(X0, Y0), K, "A", 0.9), feat(sq(X0 + 1, Y0 + 1), K, "B", 0.5)])
    r = check_both(pred, gt)
    assert r["global"][K] == (1, 1, 0)      # объект участка A забрал эталон участка B
    assert r["site"][K] == (1, 1, 0)        # здесь эталон берёт свой объект B
    metric._UTM_GEOM_CACHE.clear()
    g = metric.build_curves(pred, gt, METRIC, scope="global")[K]
    s = metric.build_curves(pred, gt, METRIC, scope="site")[K]
    assert g.hit.tolist() == [True, False] and s.hit.tolist() == [False, True]


def test_cross_site_only_foreign():
    gt = fc([feat(sq(X0, Y0), K, "B"), feat(sq(X0 + 50, Y0), K, "A")])
    pred = fc([feat(sq(X0, Y0), K, "A", 0.9)])
    r = check_both(pred, gt)
    assert r["global"][K] == (1, 0, 1) and r["site"][K] == (0, 1, 2)


def test_sub_region_and_default_source():
    gt = fc([feat(sq(X0, Y0), K, "R", sub="s1"), feat(sq(X0 + 100, Y0), K)])
    pred = fc([feat(sq(X0, Y0), K, "R", 0.8, sub="s2"),       # другой под-участок — чужой
               feat(sq(X0 + 100, Y0), K, None, 0.7),         # оба без region_name — default_source
               feat(sq(X0, Y0), K, "R", 0.6, sub="s1"),
               feat(sq(X0, Y0), K, "R", 0.5, sub="  ")])      # пустой sub = просто "R"
    r = check_both(pred, gt, taus=(0.0, 0.65))
    assert r["site"][K] == (2, 2, 0) and r["global"][K] == (2, 2, 0)


def test_cross_site_points():
    P = "finds_points"
    gt = fc([{"type": "Feature", "geometry": {"type": "Point", "coordinates": [X0, Y0]},
              "properties": {"class_name": P, "region_name": "B"}}])
    pred = fc([feat(sq(X0 - 1, Y0 - 1, 2.0), P, "A", 0.9), feat(sq(X0 - 1.2, Y0 - 1, 2.0), P, "B", 0.4)])
    r = check_both(pred, gt)
    assert r["global"][P] == (1, 1, 0) and r["site"][P] == (1, 1, 0)


def test_env_switch(monkeypatch):
    gt = fc([feat(sq(X0, Y0), K, "B")])
    pred = fc([feat(sq(X0, Y0), K, "A", 0.9)])
    assert metric.match_scope() == "global"
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "site")
    assert metric.build_curves(pred, gt, METRIC)[K].hit.tolist() == [False]
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "global")
    assert metric.build_curves(pred, gt, METRIC)[K].hit.tolist() == [True]
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "bogus")
    with pytest.raises(ValueError):
        metric.build_curves(pred, gt, METRIC)


def test_random_multi_site():
    rng = np.random.default_rng(7)
    gts, preds = [], []
    sites = ["S1", "S2", "S3", None]
    for i in range(60):
        x, y = X0 + rng.uniform(0, 120), Y0 + rng.uniform(0, 120)
        gts.append(feat(sq(x, y, rng.uniform(4, 12)), ["kurgany_tselye", "kurgany_povrezhdennye"][i % 2],
                        sites[rng.integers(4)]))
    for i in range(160):
        x, y = X0 + rng.uniform(0, 120), Y0 + rng.uniform(0, 120)
        preds.append(feat(sq(x, y, rng.uniform(4, 12)), ["kurgany_tselye", "kurgany_povrezhdennye"][i % 2],
                          sites[rng.integers(4)], float(np.round(rng.uniform(), 2))))   # есть равные уверенности
    check_both(fc(preds), fc(gts), taus=(0.0, 0.3, 0.7))


@pytest.mark.skipif(not (CROSS_PRED.exists() and CROSS_GT.exists()), reason="нет выборки соседних участков bot-8")
def test_real_neighbour_sites():
    """13 соседних участков, где найдены межучастковые пары (Гнёздово, Боровичи, Миловидово…)."""
    pred = json.loads(CROSS_PRED.read_text()); gt = json.loads(CROSS_GT.read_text())
    r = check_both(pred, gt, taus=(0.0, 0.2, 0.5))
    assert r["global"] != r["site"]          # межучастковые совпадения в выборке реально есть


@pytest.mark.skipif(not (CROSS_PRED.exists() and CROSS_GT.exists()), reason="нет выборки соседних участков bot-8")
def test_site_fast_collect_equals_reference(monkeypatch):
    """scope="site": быстрый _collect (Rust/numpy) == прежний _collect_reference, кривые побайтно."""
    pred = json.loads(CROSS_PRED.read_text()); gt = json.loads(CROSS_GT.read_text())
    fast = metric.build_curves(pred, gt, METRIC, scope="site")
    metric._UTM_GEOM_CACHE.clear()
    monkeypatch.setattr(metric, "_collect", metric._collect_reference)
    ref = metric.build_curves(pred, gt, METRIC, scope="site")
    assert list(fast) == list(ref)
    for c in ref:
        a, b = fast[c], ref[c]
        assert a.conf.tobytes() == b.conf.tobytes() and a.hit.tobytes() == b.hit.tobytes()
        assert a.pred_region == b.pred_region and a.pred_group == b.pred_group
        assert a.n_gt == b.n_gt and a.gt_regions == b.gt_regions
