"""W1-SATREC: sat_equal_fusion / sat_center_prior (flags off by default)."""
from shapely.geometry import Point

from expds import sat_recall as SR


def _f(x, y, cls, conf, grp, reg="r1"):
    return dict(geom=Point(x, y).buffer(3), class_name=cls, confidence=conf, region_name=reg, markup_type="sp_or", threshold_group=grp)


def test_fusion_mean_with_absent_members():
    feats = [_f(0, 0, "kurgany_tselye", 0.9, "sp_dfine"), _f(2, 0, "kurgany_tselye", 0.6, "sp_focal"),
             _f(500, 500, "kurgany_tselye", 0.3, "sp_focal2"), dict(_f(0, 0, "kurgany_tselye", 0.7, None), markup_type="li")]
    out = SR.sat_equal_fusion(feats)
    fused = sorted((f for f in out if f["threshold_group"] == "sp_fuse"), key=lambda f: -f["confidence"])
    assert len(fused) == 2 and len(out) == 3
    assert abs(fused[0]["confidence"] - (0.9 + 0.6) / 3) < 1e-9 and fused[0]["fuse_anchor"] == "dfine"
    assert abs(fused[1]["confidence"] - 0.3 / 3) < 1e-9


def test_center_prior_only_sat_group():
    a = dict(_f(0, 0, "kurgany_tselye", 0.8, "sp_dfine"), scene_drel=1.0)
    b = dict(_f(0, 0, "kurgany_tselye", 0.8, "sp_focal"), scene_drel=1.0)
    out = SR.sat_center_prior([a, b], {"d0": 0.5, "edge": 0.5})
    assert abs(out[0]["confidence"] - 0.4) < 1e-9 and out[1]["confidence"] == 0.8


def test_center_prior_v2_per_class_strong_and_lidar_rule():
    a = dict(_f(0, 0, "kurgany_tselye", 0.8, "sp_dfine"), scene_drel=0.75)
    b = dict(_f(0, 0, "kurgany_povrezhdennye", 0.8, "sp_dfine"), scene_drel=0.75)
    c = dict(_f(0, 0, "kurgany_tselye", 0.8, "sp_dfine", reg="r2"), scene_drel=1.0)
    li = dict(_f(0, 0, "kurgany_tselye", 0.9, None, reg="r2"), markup_type="li")
    out = SR.sat_center_prior([a, b, c, li], {"d0": 0.5, "s": 1.5, "s_by_class": {"kurgany_tselye": 0.5},
                                              "only_without_lidar": True, "nl_group": "sp_dfine_nl"})
    assert abs(out[0]["confidence"] - 0.8 * 0.75) < 1e-9 and out[0]["threshold_group"] == "sp_dfine_nl"
    assert abs(out[1]["confidence"] - 0.8 * 0.25) < 1e-9
    assert out[2] is c and out[3] is li
    z = SR.sat_center_prior([dict(c, region_name="r3")], {"d0": 0.0, "s": 1.5})
    assert z[0]["confidence"] == 0.0 and z[0]["threshold_group"] == "sp_dfine"


# ---- W1-FRZ production flags [post.w1_c1f] / [post.w1_p2f] (docs/W1_C1F_HANDOFF.md) ----
import copy
import tomllib
from pathlib import Path

import pytest

CFG = Path(__file__).resolve().parents[1] / "configs" / "v3_on.toml"


def _cfg():
    return tomllib.loads(CFG.read_text(encoding="utf-8"))


def test_final_flags_present_and_off_in_v3_on():
    cfg = _cfg()
    for k in SR.FINAL_FLAGS:
        fl = cfg["post"][k]
        assert fl["enabled"] is False and fl["groups"] == ["sp_dfine"]
        assert set(fl["thresholds_sp_dfine"]) == set(cfg["thresholds_by_markup"]["sp_dfine"])
    assert SR.final_flag_on(cfg) is None and SR.resolve_final_flag(cfg) is cfg
    assert "sat_center_prior" not in cfg["post"]
    c1 = cfg["post"]["w1_c1f"]
    assert c1["d0"] == 0.25 and set(c1["s_by_class"].values()) == {0.75} and len(c1["s_by_class"]) == 5
    assert set(cfg["post"]["w1_p2f"]["s_by_class"]) == {"gorodishcha", "arkhitektury"}


@pytest.mark.parametrize("name", ["w1_c1f", "w1_p2f"])
def test_final_flag_resolves_spec_and_only_sp_dfine_thresholds(name):
    cfg = _cfg(); before = copy.deepcopy(cfg)
    cfg["post"][name]["enabled"] = True
    r = SR.resolve_final_flag(cfg)
    fl = cfg["post"][name]
    assert r["post"]["sat_center_prior"] == {"groups": ["sp_dfine"], "d0": fl["d0"], "s": 0.0, "s_by_class": fl["s_by_class"]}
    assert r["thresholds_by_markup"]["sp_dfine"] == fl["thresholds_sp_dfine"]
    for g, d in before["thresholds_by_markup"].items():
        if g != "sp_dfine":
            assert r["thresholds_by_markup"][g] == d
    assert cfg["thresholds_by_markup"] == before["thresholds_by_markup"] and "sat_center_prior" not in cfg["post"]
    assert SR.resolve_final_flag(r) is r            # idempotent (finish after safe_finish)
    assert r["post"]["frame_prior"] == before["post"]["frame_prior"]


def test_final_flag_conflicts():
    cfg = _cfg()
    cfg["post"]["w1_c1f"]["enabled"] = cfg["post"]["w1_p2f"]["enabled"] = True
    with pytest.raises(ValueError):
        SR.resolve_final_flag(cfg)
    cfg = _cfg(); cfg["post"]["w1_c1f"]["enabled"] = True; cfg["post"]["sat_center_prior"] = {"d0": 0.5}
    with pytest.raises(ValueError):
        SR.resolve_final_flag(cfg)


def test_final_flag_prior_values_c1f():
    cfg = _cfg(); cfg["post"]["w1_c1f"]["enabled"] = True
    spec = SR.resolve_final_flag(cfg)["post"]["sat_center_prior"]
    mk = lambda cls, d, grp="sp_dfine": dict(_f(0, 0, cls, 0.8, grp), scene_drel=d)
    feats = [mk("gorodishcha", 0.2), mk("gorodishcha", 0.625), mk("kurgany_tselye", 1.0), mk("object_poly", 1.0, "sp_or"),
             _f(0, 0, "arkhitektury", 0.8, "sp_dfine")]
    out = SR.sat_center_prior(feats, spec)
    assert out[0]["confidence"] == 0.8                                       # inside d0
    assert abs(out[1]["confidence"] - 0.8 * (1 - 0.75 * 0.5)) < 1e-9          # (0.625 − 0.25)/0.75 = 0.5
    assert abs(out[2]["confidence"] - 0.8 * 0.25) < 1e-9                       # edge: 1 − 0.75
    assert out[3]["confidence"] == 0.8 and out[4]["confidence"] == 0.8         # other group / no scene_drel


def test_final_flag_p2f_leaves_other_classes():
    cfg = _cfg(); cfg["post"]["w1_p2f"]["enabled"] = True
    spec = SR.resolve_final_flag(cfg)["post"]["sat_center_prior"]
    out = SR.sat_center_prior([dict(_f(0, 0, "kurgany_tselye", 0.8, "sp_dfine"), scene_drel=1.0),
                               dict(_f(0, 0, "gorodishcha", 0.8, "sp_dfine"), scene_drel=0.5)], spec)
    assert out[0]["confidence"] == 0.8 and abs(out[1]["confidence"] - 0.8 * (1 - 1.5 * 0.5)) < 1e-9


def test_watchdog_finish_uses_flag(monkeypatch):
    from expds import watchdog
    seen = {}

    def fake(feats, cfg, isolate=None):
        seen["cfg"] = cfg
        return feats
    monkeypatch.setattr(watchdog, "_finish_v3", fake)
    cfg = _cfg()
    watchdog.finish([], cfg)
    assert seen["cfg"] is cfg
    cfg["post"]["w1_c1f"]["enabled"] = True
    watchdog.safe_finish([], cfg)
    assert seen["cfg"]["post"]["sat_center_prior"]["d0"] == 0.25
    assert seen["cfg"]["thresholds_by_markup"]["sp_dfine"] == cfg["post"]["w1_c1f"]["thresholds_sp_dfine"]


def test_sat_frame_gate_and_scene_mpix():
    from shapely.geometry import Point
    big = (0.0, 0.0, 2000.0, 2000.0)      # 16 Mpix at the equator; small box 0.04 Mpix
    small = (5000.0, 0.0, 5100.0, 100.0)
    feats = [dict(geom=Point(1900, 1000), class_name="kurgany_tselye", confidence=0.5, markup_type="sp_or", threshold_group="sp_dfine"),
             dict(geom=Point(1000, 1000), class_name="kurgany_tselye", confidence=0.5, markup_type="sp_or", threshold_group="sp_dfine"),
             dict(geom=Point(1900, 1000), class_name="kurgany_tselye", confidence=0.9, markup_type="sp_or", threshold_group="sp_dfine"),
             dict(geom=Point(1900, 1000), class_name="fortifikatsii", confidence=0.5, markup_type="sp_or", threshold_group="sp_dfine"),
             dict(geom=Point(5095, 50), class_name="kurgany_tselye", confidence=0.5, markup_type="sp_or", threshold_group="sp_dfine"),
             dict(geom=Point(9000, 9000), class_name="kurgany_tselye", confidence=0.5, markup_type="sp_or", threshold_group="sp_dfine")]
    from expds import post
    post.annotate_scene_drel(feats, {"sp_or": [big, small]})
    SR.annotate_scene_mpix(feats, {"sp_or": [big, small]})
    assert abs(feats[0]["scene_mpix"] - 16.0) < 1e-6 and feats[4]["scene_mpix"] < 0.1 and "scene_mpix" not in feats[5]
    out = SR.sat_frame_gate(feats, {"groups": ["sp_dfine"], "min_scene_mpix": 4, "x": 0.5, "t2": 0.7})
    assert [id(f) for f in out] == [id(f) for f in feats[1:]]   # only the edge, low-score kt on the big scene goes


def test_center_prior_then_frame_prior():
    spec = {"groups": ["sp_dfine"], "d0": 0.25, "s": 0.0, "s_by_class": {"kurgany_tselye": 0.75, "gorodishcha": 0.75},
            "frame_prior": {"classes": ["kurgany_tselye"], "d0": 0.5, "edge": 0.5}}
    feats = [dict(_f(0, 0, "kurgany_tselye", 0.8, "sp_dfine"), scene_drel=1.0),
             dict(_f(0, 0, "gorodishcha", 0.8, "sp_dfine"), scene_drel=1.0),
             dict(_f(0, 0, "kurgany_tselye", 0.8, "sp_or"), scene_drel=1.0)]
    out = SR.sat_center_prior(feats, spec)
    assert abs(out[0]["confidence"] - 0.8 * 0.25 * 0.5) < 1e-9    # C1F edge 1 − 0.75, then FP1 edge 0.5
    assert abs(out[1]["confidence"] - 0.8 * 0.25) < 1e-9          # gorodishcha: not an FP1 class
    assert out[2]["confidence"] == 0.8                            # other group untouched


def test_sat_frame_gate_t2_cls():
    from shapely.geometry import Point
    from expds import post
    big = (0.0, 0.0, 2000.0, 2000.0)
    feats = [dict(geom=Point(1900, 1000), class_name=c, confidence=0.6, markup_type="sp_or", threshold_group="sp_dfine")
             for c in ("kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha")]
    post.annotate_scene_drel(feats, {"sp_or": [big]})
    SR.annotate_scene_mpix(feats, {"sp_or": [big]})
    out = SR.sat_frame_gate(feats, {"min_scene_mpix": 1, "x": 0.5, "t2_cls": {"kurgany_tselye": 0.7, "gorodishcha": 0.5}})
    assert [f["class_name"] for f in out] == ["kurgany_povrezhdennye", "gorodishcha"]   # kt < 0.7 dropped; kp no t2 → kept
    out = SR.sat_frame_gate(feats, {"min_scene_mpix": 1, "x": 0.5, "t2": 0.65, "t2_cls": {"gorodishcha": 0.5}})
    assert [f["class_name"] for f in out] == ["gorodishcha"]
