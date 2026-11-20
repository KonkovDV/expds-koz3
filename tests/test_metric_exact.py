"""Точный подбор порогов: один проход сопоставления даёт ровно тот же балл,
что полный официальный расчёт, при любых порогах. Нужен EXPDS_METRIC."""
from __future__ import annotations

import random

import numpy as np
import pytest
from pyproj import Transformer

from expds import metric

T = Transformer.from_crs(32636, 3857, always_xy=True)


def _circle(cx, cy, r, n=20):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = [list(T.transform(cx + r * np.cos(a), cy + r * np.sin(a))) for a in ang]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


def _f(geom, cls, region, conf=None):
    props = {"class_name": cls, "region_name": region}
    if conf is not None:
        props["confidence"] = conf
    return {"type": "Feature", "geometry": geom, "properties": props}


def make_case(seed: int):
    """Курганы/городища/архитектура (полигоны), находки (точки), дубли, ложные, равные уверенности."""
    rng = random.Random(seed)
    gt, pr = [], []
    for region, (x0, y0) in {"region_0001": (400_000, 6_000_000), "region_0002": (420_000, 6_050_000),
                             "region_0003": (380_000, 5_950_000)}.items():
        for _ in range(25):
            cls = rng.choice(["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "arkhitektury"])
            cx, cy, r = x0 + rng.uniform(0, 2000), y0 + rng.uniform(0, 2000), rng.uniform(3, 12)
            gt.append(_f(_circle(cx, cy, r), cls, region))
            if rng.random() < 0.75:
                conf = round(rng.uniform(0.2, 1.0), 2)
                pr.append(_f(_circle(cx + rng.uniform(-r, r), cy + rng.uniform(-r, r), r * rng.uniform(0.5, 1.6)),
                             cls, region, conf))
                if rng.random() < 0.2:
                    pr.append(_f(_circle(cx, cy, r), cls, region, round(conf * 0.8, 2)))
        for _ in range(15):
            cx, cy = x0 + rng.uniform(0, 500), y0 + rng.uniform(0, 500)
            gt.append({"type": "Feature", "properties": {"class_name": "finds_points", "region_name": region},
                       "geometry": {"type": "Point", "coordinates": list(T.transform(cx, cy))}})
            if rng.random() < 0.7:
                pr.append(_f(_circle(cx + rng.uniform(-2.5, 2.5), cy + rng.uniform(-2.5, 2.5), 0.5),
                             "finds_points", region, round(rng.uniform(0, 1), 1)))
        for _ in range(30):
            cls = rng.choice(["kurgany_tselye", "kurgany_povrezhdennye", "finds_points", "fortifikatsii"])
            pr.append(_f(_circle(x0 + rng.uniform(0, 2000), y0 + rng.uniform(0, 2000), rng.uniform(1, 8)),
                         cls, region, round(rng.uniform(0, 0.7), 2)))
    return {"type": "FeatureCollection", "features": pr}, {"type": "FeatureCollection", "features": gt}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_pass_equals_official(seed, official_metric):
    if not official_metric:
        pytest.skip("нет EXPDS_METRIC")
    pred, gt = make_case(seed)
    curves = metric.build_curves(pred, gt, official_metric)
    rng = random.Random(100 + seed)
    for _ in range(5):
        taus = {c: rng.choice([0.0, 0.2, 0.35, 0.5, 0.7, 0.9]) for c in curves}
        res = metric.verify(pred, gt, taus, official_metric, curves)
        assert res["abs_diff"] < 1e-12, res
    sw = metric.sweep(pred, gt, official_metric, n_boot=50)
    res = metric.verify(pred, gt, sw["thresholds"], official_metric, curves)
    assert res["abs_diff"] < 1e-12
    assert res["official"] >= metric.score(pred, gt, official_metric)["final_score"] - 1e-12
    assert abs(res["official"] - sw["estimated_score"]) < 1e-6
    assert set(sw["bootstrap"]) <= set(sw["thresholds"])


def test_group_thresholds_beat_single_and_match_official(official_metric):
    if not official_metric:
        pytest.skip("нет EXPDS_METRIC")
    pred, gt = make_case(1)
    regions = sorted({f["properties"]["region_name"] for f in pred["features"]})
    half = set(regions[: len(regions) // 2])
    feats = []
    for f in pred["features"]:
        pr = dict(f["properties"])
        if pr["region_name"] in half:
            pr["markup_type"] = "li"
        else:
            pr["markup_type"] = "sp_or"
            pr["confidence"] = round(pr["confidence"] * 0.5, 3)  # другая шкала уверенности
        feats.append({**f, "properties": pr})
    pred = {**pred, "features": feats}
    curves = metric.build_curves(pred, gt, official_metric)
    single = metric.sweep(pred, gt, official_metric)["estimated_score"]
    res = metric.sweep_groups(curves)
    taus: dict = {}
    for cls, v in res.items():
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
    est = metric.official_weighted(res)
    official = metric.score(metric.apply_group_thresholds(pred, taus, official_metric), gt,
                            official_metric)["final_score"]
    assert abs(est - official) < 1e-6
    assert official >= single - 1e-12


def test_paired_bootstrap_zero_for_same_model_and_sign_for_worse(official_metric):
    if not official_metric:
        pytest.skip("нет EXPDS_METRIC")
    pred, gt = make_case(0)
    sw = metric.sweep(pred, gt, official_metric)
    same = metric.paired_bootstrap(sw["_curves"], sw["_curves"], sw["thresholds"], sw["thresholds"], 200)
    assert same["score"]["p5"] == same["score"]["p95"] == 0.0
    worse = {**pred, "features": pred["features"][::2]}
    sw_b = metric.sweep(worse, gt, official_metric)
    res = metric.paired_bootstrap(sw["_curves"], sw_b["_curves"], sw["thresholds"], sw_b["thresholds"], 200)
    assert res["score"]["mean"] > 0 and res["score"]["p_gt0"] > 0.9
