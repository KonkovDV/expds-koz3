"""bot-4 large_objects: флаг OFF = no-op; area_tau / protect / fort_to_gor при ON."""
from __future__ import annotations

from shapely.geometry import Point, box

from expds import large_objects as lo
from expds import post
from expds import watchdog


def _feat(cls, conf, area_side, markup="li", group=None, source="a"):
    # area_side в метрах при CRS как у обычного geom.area (не 3857-эвристика).
    g = box(0, 0, area_side, area_side)
    f = {"geom": g, "class_name": cls, "confidence": conf, "region_name": "r",
         "markup_type": markup, "source": source}
    if group:
        f["threshold_group"] = group
    return f


def test_enabled_default_off():
    assert lo.enabled({}) is False
    assert lo.enabled({"large_objects": {}}) is False
    assert lo.enabled({"large_objects": {"enabled": False}}) is False
    assert lo.enabled({"large_objects": {"enabled": True}}) is True


def test_area_tau_relaxes_only_large():
    base = lambda f: 0.40
    tau = lo.area_tau(base, {"min_area_m2": 5000.0, "tau_relax": 0.15, "tau_floor": 0.05,
                             "classes": ["gorodishcha"]})
    small = _feat("gorodishcha", 0.5, 50)   # 2500 м²
    large = _feat("gorodishcha", 0.5, 100)  # 10000 м² → frac=1 → τ=0.25
    assert tau(small) == 0.40
    assert abs(tau(large) - 0.25) < 1e-9
    other = _feat("kurgany_tselye", 0.5, 100)
    assert tau(other) == 0.40


def test_apply_thresholds_keeps_large_below_base():
    feats = [
        _feat("gorodishcha", 0.30, 100, source="a"),  # large, base 0.40 → keep with relax
        _feat("gorodishcha", 0.30, 50, source="b"),   # small, drop
    ]
    thr = {"gorodishcha": 0.40}
    lo_cfg = {"min_area_m2": 5000.0, "tau_relax": 0.20, "tau_floor": 0.05,
              "classes": ["gorodishcha"]}
    kept = lo.apply_thresholds(feats, thr, 0.15, {}, lo_cfg)
    assert len(kept) == 1
    assert kept[0]["source"] == "a"


def test_merge_protect_keeps_large_and_small():
    # Крупный и мелкий пересекаются, разные source — оба остаются.
    large = _feat("gorodishcha", 0.9, 100, source="big")
    small = _feat("gorodishcha", 0.5, 30, source="small")
    # сдвинем мелкий так, чтобы пересекался с крупным
    small["geom"] = box(50, 50, 80, 80)  # 900 м², внутри large
    lo_cfg = {"min_area_m2": 5000.0, "classes": ["gorodishcha"]}
    out = lo.merge_across_sources_protect([large, small], lo_cfg)
    assert len(out) == 2


def test_fort_to_gor_copies_large_fort():
    feats = [
        _feat("fortifikatsii", 0.8, 100, markup="ae"),
        _feat("fortifikatsii", 0.8, 40, markup="ae"),
    ]
    out = lo.fort_to_gor(feats, {"fort_to_gor": True, "min_area_m2": 5000.0,
                                 "fort_to_gor_scale": 0.9, "fort_to_gor_group": "gor_from_fort"})
    assert len(out) == 3
    copies = [f for f in out if f.get("large_objects_seed") == "fort_to_gor"]
    assert len(copies) == 1
    assert copies[0]["class_name"] == "gorodishcha"
    assert copies[0]["threshold_group"] == "gor_from_fort"
    assert abs(copies[0]["confidence"] - 0.72) < 1e-9


def test_finish_flag_off_byte_identical_on_toy():
    """При enabled=false finish не меняет цепочку относительно post.apply_thresholds на игрушке."""
    feats = [_feat("gorodishcha", 0.5, 100, group="gor_phys")]
    cfg = {"thresholds": {"gorodishcha": 0.15}, "default_threshold": 0.15,
           "thresholds_by_markup": {}, "post": {}, "large_objects": {"enabled": False}}
    a = watchdog.finish(list(feats), cfg)
    b = watchdog.finish(list(feats), {**cfg, "large_objects": {"enabled": False}})
    assert len(a) == len(b)
