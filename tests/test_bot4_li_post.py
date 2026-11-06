""" li_post: master OFF = no-op; class_tau floors; self_nms radius."""
from __future__ import annotations

import pytest
from shapely.geometry import Point

from expds import li_post as lp
from expds import watchdog


def _feat(cls, conf, x, y, markup="li", group="li_cnx"):
    f = {
        "geom": Point(x, y).buffer(4.0),
        "class_name": cls,
        "confidence": conf,
        "region_name": "r",
        "markup_type": markup,
        "threshold_group": group,
    }
    return f


def test_default_toml_li_post_off():
    import tomllib
    from pathlib import Path
    cfg = tomllib.loads(Path("configs/default.toml").read_text(encoding="utf-8"))
    assert cfg["li_post"]["enabled"] is False
    assert cfg["li_post"]["class_tau"] is False
    assert cfg["li_post"]["self_nms"] is False


def test_enabled_and_flag():
    assert lp.enabled({}) is False
    assert lp.flag({"li_post": {"enabled": False, "class_tau": True}}, "class_tau") is False
    assert lp.flag({"li_post": {"enabled": True}}, "class_tau") is False
    assert lp.flag({"li_post": {"enabled": True, "class_tau": True}}, "class_tau") is True


def test_apply_noop_when_off():
    feats = [_feat("kurgany_povrezhdennye", 0.20, 0, 0)]
    out = lp.apply(feats, {"li_post": {"enabled": False, "class_tau": True}})
    assert out is feats or out == feats


def test_class_tau_drops_below_floor_only_li():
    feats = [
        _feat("kurgany_povrezhdennye", 0.20, 0, 0),          # drop
        _feat("kurgany_povrezhdennye", 0.35, 10, 0),         # keep
        _feat("kurgany_tselye", 0.25, 20, 0),                # drop
        dict(_feat("kurgany_povrezhdennye", 0.10, 30, 0), markup_type="ae", threshold_group="ae"),  # non-li keep
    ]
    floors = {"kurgany_povrezhdennye": 0.30, "kurgany_tselye": 0.30}
    out = lp.apply_class_tau(feats, floors)
    assert len(out) == 2
    assert out[0]["confidence"] == 0.35
    assert out[1]["markup_type"] == "ae"


def test_self_nms_keeps_higher_conf_within_radius():
    feats = [
        _feat("kurgany_povrezhdennye", 0.9, 0, 0),
        _feat("kurgany_povrezhdennye", 0.5, 3, 0),   # within 8 m → drop
        _feat("kurgany_povrezhdennye", 0.4, 20, 0),  # outside → keep
        _feat("kurgany_tselye", 0.8, 1, 0),          # other class → keep
    ]
    out = lp.self_nms(feats, 8.0)
    confs = sorted(f["confidence"] for f in out)
    assert confs == [0.4, 0.8, 0.9]


def test_finish_off_byte_path_no_drop():
    feats = [_feat("kurgany_povrezhdennye", 0.20, 0, 0)]
    cfg = {
        "thresholds": {"kurgany_povrezhdennye": 0.10},
        "default_threshold": 0.10,
        "thresholds_by_markup": {},
        "li_post": {"enabled": False, "class_tau": True, "self_nms": True,
                    "class_tau_floor": {"kurgany_povrezhdennye": 0.50}, "self_nms_m": 8.0},
        "post": {},
    }
    out = watchdog.finish(feats, cfg)
    assert len(out) == 1


def test_finish_class_tau_on_drops():
    feats = [
        _feat("kurgany_povrezhdennye", 0.20, 0, 0),
        _feat("kurgany_povrezhdennye", 0.40, 50, 0),
    ]
    cfg = {
        "thresholds": {"kurgany_povrezhdennye": 0.10},
        "default_threshold": 0.10,
        "thresholds_by_markup": {},
        "li_post": {"enabled": True, "class_tau": True, "self_nms": False,
                    "class_tau_floor": {"kurgany_povrezhdennye": 0.30}},
        "post": {},
    }
    out = watchdog.finish(feats, cfg)
    assert len(out) == 1
    assert out[0]["confidence"] == 0.40
