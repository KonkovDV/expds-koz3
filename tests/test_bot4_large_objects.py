"""bot-4 large_objects: мастер OFF = no-op; sub-flags независимо; area_tau/protect/fort_to_gor."""
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


def test_flag_requires_master_and_defaults_false():
    cfg_off = {"large_objects": {"enabled": False, "coarse_hillfort": True}}
    assert lo.flag(cfg_off, "coarse_hillfort") is False
    cfg_on = {"large_objects": {"enabled": True}}
    for name in ("area_tau", "protect_merge", "shape_cores", "fort_to_gor", "coarse_hillfort",
                 "dfine_ae_gor_area_tau"):
        assert lo.flag(cfg_on, name) is False
    cfg_one = {"large_objects": {"enabled": True, "fort_to_gor": True}}
    assert lo.flag(cfg_one, "fort_to_gor") is True
    assert lo.flag(cfg_one, "coarse_hillfort") is False


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
    lo_cfg = {"area_tau": True, "min_area_m2": 5000.0, "tau_relax": 0.20, "tau_floor": 0.05,
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
                                 "fort_to_gor_scale": 0.9, "fort_to_gor_group": "gor_from_fort",
                                 "fort_to_gor_min_conf": 0.5})
    assert len(out) == 3
    copies = [f for f in out if f.get("large_objects_seed") == "fort_to_gor"]
    assert len(copies) == 1
    assert copies[0]["class_name"] == "gorodishcha"
    assert copies[0]["threshold_group"] == "gor_from_fort"
    assert abs(copies[0]["confidence"] - 0.72) < 1e-9
    # keep-original-label: fort остаётся
    assert sum(1 for f in out if f["class_name"] == "fortifikatsii") == 2


def test_fort_to_gor_skips_overlap_and_low_conf():
    fort = _feat("fortifikatsii", 0.8, 100, markup="ae", source="f")
    gor = _feat("gorodishcha", 0.9, 100, markup="li", source="g")
    # fort полностью пересекается с gor → копии нет
    out = lo.fort_to_gor([fort, gor], {"fort_to_gor": True, "min_area_m2": 5000.0,
                                       "fort_to_gor_scale": 0.8, "fort_to_gor_min_conf": 0.5})
    assert not any(f.get("large_objects_seed") == "fort_to_gor" for f in out)
    # низкий conf → копии нет
    weak = _feat("fortifikatsii", 0.3, 100, markup="ae", source="w")
    out2 = lo.fort_to_gor([weak], {"fort_to_gor": True, "min_area_m2": 5000.0,
                                   "fort_to_gor_min_conf": 0.5})
    assert not any(f.get("large_objects_seed") == "fort_to_gor" for f in out2)


def test_finish_flag_off_byte_identical_on_toy():
    """При enabled=false finish не меняет цепочку относительно post.apply_thresholds на игрушке."""
    feats = [_feat("gorodishcha", 0.5, 100, group="gor_phys")]
    cfg = {"thresholds": {"gorodishcha": 0.15}, "default_threshold": 0.15,
           "thresholds_by_markup": {}, "post": {}, "large_objects": {"enabled": False}}
    a = watchdog.finish(list(feats), cfg)
    b = watchdog.finish(list(feats), {**cfg, "large_objects": {"enabled": False}})
    assert len(a) == len(b)


def test_finish_master_on_subs_off_same_as_off():
    """enabled=true без sub-flags = тот же путь, что enabled=false (на игрушке)."""
    feats = [_feat("gorodishcha", 0.5, 100, group="gor_phys"),
             _feat("fortifikatsii", 0.8, 100, markup="ae")]
    base = {"thresholds": {"gorodishcha": 0.15, "fortifikatsii": 0.15},
            "default_threshold": 0.15, "thresholds_by_markup": {}, "post": {}}
    off = watchdog.finish(list(feats), {**base, "large_objects": {"enabled": False}})
    on_empty = watchdog.finish(list(feats), {**base, "large_objects": {"enabled": True}})
    assert len(off) == len(on_empty)
    assert [f["class_name"] for f in off] == [f["class_name"] for f in on_empty]


def test_dfine_ae_gor_area_tau_only_large_ae_gor():
    """Fix C: абсолютный τ_large только ae+dfine_ae+gorodishcha+box_area≥min; li не трогаем."""
    base = lambda f: 1.01
    lo_cfg = {"dfine_ae_gor_area_tau": True, "dfine_ae_gor_tau_large": 0.20,
              "min_area_m2": 5000.0,
              "dfine_ae_gor_classes": ["gorodishcha"],
              "dfine_ae_gor_markups": ["ae"],
              "dfine_ae_gor_groups": ["dfine_ae", "ae_dfine"]}
    tau = lo.dfine_ae_gor_area_tau(base, lo_cfg)
    large_ae = _feat("gorodishcha", 0.25, 100, markup="ae", group="dfine_ae")
    large_ae["box_area_m2"] = 10000.0
    small_ae = _feat("gorodishcha", 0.25, 50, markup="ae", group="dfine_ae")
    small_ae["box_area_m2"] = 2500.0
    li_large = _feat("gorodishcha", 0.25, 100, markup="li", group="gor_phys")
    li_large["box_area_m2"] = 10000.0
    fort = _feat("fortifikatsii", 0.25, 100, markup="ae", group="dfine_ae")
    fort["box_area_m2"] = 10000.0
    assert abs(tau(large_ae) - 0.20) < 1e-9
    assert tau(small_ae) == 1.01
    assert tau(li_large) == 1.01
    assert tau(fort) == 1.01
    # alias ae_dfine
    alias = _feat("gorodishcha", 0.25, 100, markup="ae", group="ae_dfine")
    alias["box_area_m2"] = 10000.0
    assert abs(tau(alias) - 0.20) < 1e-9


def test_apply_thresholds_fix_c_keeps_mid_conf_large_ae():
    """При gor=1.01 Fix C пропускает крупный ae dfine_ae с conf∈[τ_large, 1.01)."""
    feats = [
        _feat("gorodishcha", 0.25, 100, markup="ae", group="dfine_ae", source="keep"),
        _feat("gorodishcha", 0.10, 100, markup="ae", group="dfine_ae", source="drop_low"),
        _feat("gorodishcha", 0.25, 100, markup="li", group="gor_phys", source="li_untouched"),
    ]
    feats[0]["box_area_m2"] = 12000.0
    feats[1]["box_area_m2"] = 12000.0
    feats[2]["box_area_m2"] = 12000.0
    by_markup = {"dfine_ae": {"gorodishcha": 1.01}, "ae": {"gorodishcha": 1.01}}
    lo_cfg = {"dfine_ae_gor_area_tau": True, "dfine_ae_gor_tau_large": 0.20,
              "min_area_m2": 5000.0}
    kept = lo.apply_thresholds(feats, {"gorodishcha": 1.01}, 0.15, by_markup, lo_cfg)
    srcs = {f["source"] for f in kept}
    assert "keep" in srcs
    assert "drop_low" not in srcs
    # li stays at 1.01 → dropped (conf 0.25 < 1.01) — correct, Fix C does not touch li
    assert "li_untouched" not in srcs


def test_area_m2_prefers_box_area_over_circle_geom():
    """Круг r≈4 м → ~50 м²; box_area_m2 побеждает для Fix C."""
    f = _feat("gorodishcha", 0.5, 8, markup="ae", group="dfine_ae")  # geom ~64 м²
    f["box_area_m2"] = 15000.0
    assert abs(lo.area_m2(f) - 15000.0) < 1e-6


def test_cluster_circles_preserves_max_box_area():
    from shapely.geometry import box as bx
    from expds import sp_dfine as S
    d1 = {"geom": bx(0, 0, 10, 10), "class_name": "gorodishcha", "confidence": 0.4,
          "region_name": "r", "model": 0, "source": "a", "original_crs": "EPSG:32637",
          "box_area_m2": 8000.0}
    d2 = {"geom": bx(1, 1, 11, 11), "class_name": "gorodishcha", "confidence": 0.3,
          "region_name": "r", "model": 0, "source": "a", "original_crs": "EPSG:32637",
          "box_area_m2": 12000.0}
    out = S.cluster_circles([d1, d2], radius_m=4.0, merge_m=50.0)
    assert len(out) == 1
    assert abs(out[0]["box_area_m2"] - 12000.0) < 1e-6


def test_finish_fix_c_flag_fires():
    """enabled + dfine_ae_gor_area_tau: крупный ae dfine_ae с conf=0.25 проходит при base 1.01."""
    f = _feat("gorodishcha", 0.25, 100, markup="ae", group="dfine_ae")
    f["box_area_m2"] = 10000.0
    cfg = {
        "thresholds": {"gorodishcha": 1.01},
        "default_threshold": 0.15,
        "thresholds_by_markup": {"dfine_ae": {"gorodishcha": 1.01}},
        "post": {},
        "large_objects": {
            "enabled": True,
            "dfine_ae_gor_area_tau": True,
            "dfine_ae_gor_tau_large": 0.20,
            "min_area_m2": 5000.0,
        },
    }
    out = watchdog.finish([f], cfg)
    assert len(out) == 1
    # OFF → drop
    cfg_off = {**cfg, "large_objects": {"enabled": True, "dfine_ae_gor_area_tau": False}}
    out_off = watchdog.finish([dict(f)], cfg_off)
    assert len(out_off) == 0
