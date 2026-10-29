"""bot-2: почему A≡B на PC — синтетические D-FINE в finish/writer/score path.

Гипотеза (b): если dets доходят до state с drop_groups=[], они появляются в GeoJSON
как markup_type=sp_or и портят precision (F1 ≠). Точное равенство A≡B на PC значит,
что D-FINE в scored output не попал (0 features), а не «13 FP на 191 м игнорируются».
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import Point

from expds import post, sp_dfine, watchdog, writer


def _circle(x, y, r, **props):
    return dict(geom=Point(x, y).buffer(r), **props)


def _cfg_thr(sp_dfine_tau=0.55, sp_or_tau=0.3):
    return {
        "thresholds": {"kurgany_povrezhdennye": 0.2},
        "default_threshold": 0.15,
        "thresholds_by_markup": {
            "sp_dfine": {"kurgany_povrezhdennye": sp_dfine_tau},
            "sp_or": {"kurgany_povrezhdennye": sp_or_tau},
        },
        "post": {},  # без micro_circles — проще считать площади
    }


def test_synthetic_dfine_reaches_geojson_as_sp_or():
    """D-FINE (threshold_group=sp_dfine) переживает finish и пишется как markup_type=sp_or."""
    unet = _circle(1e6, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.9,
                   region_name="r1", markup_type="sp_or", threshold_group="sp_or")
    # 191 м восточнее — как FP на mock (не пересекается с U-Net)
    dfine = _circle(1e6 + 191, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.65,
                    region_name="r1", markup_type="sp_or", threshold_group="sp_dfine")
    cfg = _cfg_thr()
    out = watchdog.finish([unet, dfine], cfg)
    assert len(out) == 2, f"оба должны пройти порог, got {len(out)}"
    groups = {f.get("threshold_group") for f in out}
    assert groups == {"sp_or", "sp_dfine"}
    fc = writer.to_feature_collection(out)
    props = [f["properties"] for f in fc["features"]]
    assert all(p.get("markup_type") == "sp_or" for p in props)
    # writer НЕ пишет threshold_group — скорер видит оба как sp_or
    assert all("threshold_group" not in p for p in props)
    assert len(fc["features"]) == 2


def test_far_dfine_fp_increases_sp_or_pred_count():
    """Лишние D-FINE FP увеличивают число sp_or в GeoJSON → precision/F1 должны сдвинуться.

    Точное A≡B (δ=0 по 6 знакам) на PC несовместимо с попаданием 13 FP в выдачу.
    """
    unet = [
        _circle(1e6 + 20 * i, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.9,
                region_name="r1", markup_type="sp_or", threshold_group="sp_or")
        for i in range(3)
    ]
    dfine = [
        _circle(1e6 + 200 + 20 * i, 7e6 + 200, 4.0, class_name="kurgany_povrezhdennye",
                confidence=0.65, region_name="r1", markup_type="sp_or", threshold_group="sp_dfine")
        for i in range(13)
    ]
    cfg = _cfg_thr()
    only_unet = writer.to_feature_collection(watchdog.finish(unet, cfg))
    with_dfine = writer.to_feature_collection(watchdog.finish(unet + dfine, cfg))
    assert len(only_unet["features"]) == 3
    assert len(with_dfine["features"]) == 16
    # merge_within_markup не схлопывает далёкие круги
    assert len(with_dfine["features"]) - len(only_unet["features"]) == 13


def test_empty_dfine_keeps_unet_only_like_ab_identity():
    """Паттерн A≡B: region_feats=[] (UNAVAILABLE / thr / load fail) → только U-Net."""
    unet = [
        _circle(1e6, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.9,
                region_name="r1", markup_type="sp_or", threshold_group="sp_or")
    ]
    cfg = _cfg_thr()
    a = writer.to_feature_collection(watchdog.finish(unet + [], cfg))  # A без dets
    b = writer.to_feature_collection(watchdog.finish(unet, cfg))       # B
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_apply_thresholds_uses_sp_dfine_group_not_sp_or():
    """Порог группы sp_dfine (0.55), не sp_or (0.3): conf 0.4 отсекается, 0.65 проходит."""
    low = _circle(1e6, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.40,
                  region_name="r1", markup_type="sp_or", threshold_group="sp_dfine")
    hi = _circle(1e6 + 50, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.65,
                 region_name="r1", markup_type="sp_or", threshold_group="sp_dfine")
    cfg = _cfg_thr(sp_dfine_tau=0.55, sp_or_tau=0.3)
    out = post.apply_thresholds([low, hi], cfg["thresholds"], 0.15, cfg["thresholds_by_markup"])
    assert len(out) == 1 and out[0]["confidence"] == 0.65


def test_count_would_pass_thr_matches_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPDS_DFINE_DUMP", str(tmp_path))
    feats = [
        _circle(1e6, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.56,
                region_name="r1", markup_type="sp_or", threshold_group="sp_dfine"),
        _circle(1e6 + 10, 7e6, 4.0, class_name="kurgany_povrezhdennye", confidence=0.40,
                region_name="r1", markup_type="sp_or", threshold_group="sp_dfine"),
    ]
    cfg = _cfg_thr(sp_dfine_tau=0.55)
    assert sp_dfine.count_would_pass_thr(feats, cfg) == 1


def test_stages_jsonl_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPDS_DFINE_DUMP", str(tmp_path))
    sp_dfine.append_stage(tmp_path, {
        "region": "r1", "n_raw_boxes": 482, "n_ellipses": 482, "n_features": 50,
        "n_would_pass_thr": 13, "n_paths": 1, "n_paths_done": 1,
    })
    sp_dfine.append_stage(tmp_path, {
        "region": "r2", "n_raw_boxes": 0, "n_ellipses": 0, "n_features": 0,
        "n_would_pass_thr": 0, "n_paths": 1, "n_paths_done": 0, "skip_reason": "soft_ceiling",
    })
    rows = [json.loads(l) for l in (tmp_path / "stages.jsonl").read_text().splitlines()]
    path = sp_dfine.write_stages_summary(tmp_path, rows, n_models=1, thresholds_ready=True)
    data = json.loads(path.read_text())
    assert data["totals"]["n_raw_boxes"] == 482
    assert data["totals"]["n_would_pass_thr"] == 13
    assert data["n_models"] == 1
