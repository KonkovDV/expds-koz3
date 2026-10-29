"""Арбитр fort→gor и грубый проход по стеку (expds/gorfort.py) на синтетике."""
import math

import numpy as np

from expds import gorfort as gf


def _sq(x, y, s, cls, grp, conf, reg="r1"):
    c = [[x - s, y - s], [x + s, y - s], [x + s, y + s], [x - s, y + s], [x - s, y - s]]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [c]},
            "properties": {"class_name": cls, "threshold_group": grp, "confidence": conf, "region_name": reg, "markup_type": "li"}}


def _fc():
    x0, y0 = 3_540_000.0, 7_317_000.0   # ~55° с. ш.: 1 м на местности ≈ 1,74 ед. 3857
    ring = [_sq(x0 + 80 * math.cos(a), y0 + 80 * math.sin(a), 6, gf.FORT, "li_fort", 0.99) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    lone = [_sq(x0 + 5000, y0, 6, gf.FORT, "li_fort", 0.995)]
    gor = [_sq(x0 + 10, y0 + 10, 4, gf.GOR, "li_mask", 0.5)]
    noise = [_sq(x0 - 9000, y0, 6, gf.FORT, "ae_fort", 0.5)]          # ниже ae_floor — отброшен
    return {"type": "FeatureCollection", "features": ring + lone + gor + noise}


def test_table_drops_low_ae_and_measures_in_metres():
    t = gf.table(_fc())
    assert len(t["cls"]) == 10
    k = t["k"][0]
    assert 0.55 < k < 0.6
    assert abs(t["area"][0] - (12 * k) ** 2) < 1e-6 * t["area"][0] + 1e-3


def test_ring_of_forts_scores_above_a_lone_fort_and_clusters_into_one_coarse_candidate():
    t = gf.table(_fc())
    q, F = gf.fort_features(t)
    s = gf.rule_score_fort(F)
    lone = int(np.argmax(F["conf"]))
    assert s[lone] < np.median(s)
    cl = gf.clusters(t, gf.GorFortParams(link_m=80.0))
    assert len(cl) == 1 and len(cl[0]) == 9               # кольцо + городище; одиночка не скопление
    cx, cy, F2 = gf.cluster_features(t, cl)
    assert abs(cx[0] - 3_540_000.0) < 30 and abs(cy[0] - 7_317_000.0) < 30   # центр скопления — внутри кольца
    assert F2["n_fort_li_fort"][0] == 8 and F2["n_goro_li_mask"][0] == 1


def test_circles_and_nms():
    out = gf.circles(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.5, 0.5]), np.array([0.9, 0.8]),
                     np.array(["r", "r"], dtype=object), "fort2gor")
    assert out[0]["properties"]["threshold_group"] == "fort2gor" and out[0]["properties"]["class_name"] == gf.GOR
    keep = gf.nms(np.array([0.0, 10.0, 1000.0]), np.zeros(3), np.ones(3), np.array([0.5, 0.9, 0.1]),
                  np.array(["r", "r", "r"], dtype=object), 60.0)
    assert keep.tolist() == [False, True, True]
