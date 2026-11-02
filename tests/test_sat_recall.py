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
    """W4 port: finish/safe_finish resolve the flag; prior acts on sp_dfine before apply_thresholds with the flag's τ."""
    from expds import post, watchdog
    seen = {}
    real = post.apply_thresholds

    def spy(feats, thr, default, by_markup):
        seen["by"] = by_markup
        seen["conf"] = [f["confidence"] for f in feats if f.get("threshold_group") == "sp_dfine"]
        return real(feats, thr, default, by_markup)
    monkeypatch.setattr(post, "apply_thresholds", spy)
    cfg = _cfg()
    feat = lambda: [dict(_f(0, 0, "gorodishcha", 0.8, "sp_dfine"), scene_drel=1.0)]
    watchdog.finish(feat(), cfg)
    assert seen["conf"] == [0.8] and seen["by"]["sp_dfine"]["gorodishcha"] == 0.68847
    cfg["post"]["w1_c1f"]["enabled"] = True
    for fn in (watchdog.finish, watchdog.safe_finish):
        seen.clear()
        fn(feat(), cfg)
        assert abs(seen["conf"][0] - 0.8 * 0.25) < 1e-9          # d0 0.25, s 0.75, scene edge -> m 0.25
        assert seen["by"]["sp_dfine"] == cfg["post"]["w1_c1f"]["thresholds_sp_dfine"]


def test_sp_dfine_phase_annotates_scene_drel_only_under_flag(monkeypatch):
    """W4: D-FINE features are created after optical_feats; under a W1 flag they must get scene_drel too."""
    from types import SimpleNamespace
    from expds import pipeline, sp_dfine

    def fake_region_feats(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = [object()]
        return [_f(0, 0, "gorodishcha", 0.8, "sp_dfine")], True
    monkeypatch.setattr(sp_dfine, "region_feats", fake_region_feats)
    monkeypatch.setattr(pipeline, "_with_scene_drel", lambda reg, cfg, feats: [dict(f, scene_drel=0.9) for f in feats])
    monkeypatch.setattr(pipeline, "_soft_ok", lambda *a, **k: True)
    reg = SimpleNamespace(name="r1", rasters={"sp_or": ["x.tif"]}, utm_json=None)
    guard = SimpleNamespace(ok=lambda s: True, elapsed=lambda: 0.0)
    for on in (False, True):
        cfg = _cfg()
        cfg["sp_dfine"] = dict(cfg["sp_dfine"], drop_groups=[])
        cfg["post"]["w1_c1f"]["enabled"] = on
        got = []
        st = SimpleNamespace(add=got.extend, snapshot=lambda: list(got), replace=lambda x: None)
        pipeline._sp_dfine_phase([reg], cfg, guard, st, Path("."))
        assert len(got) == 1 and (("scene_drel" in got[0]) == on)
