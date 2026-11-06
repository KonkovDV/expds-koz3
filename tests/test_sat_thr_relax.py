"""[sat_thr_relax] — scale sp_* thresholds_by_markup (default OFF)."""
from __future__ import annotations

import pytest
from shapely.geometry import Point

from expds import post
from expds.watchdog import finish


BY = {
    "sp_or": {"gorodishcha": 0.99, "kurgany_povrezhdennye": 0.5},
    "sp_focal": {"kurgany_povrezhdennye": 0.3645, "kurgany_tselye": 0.4},
    "sp_focal2": {"kurgany_povrezhdennye": 0.3},
    "sp_arch": {"arkhitektury": 0.55},
    "ae_or": {"kurgany_tselye": 0.7},  # not a sat group — must stay
}


def test_scale_sat_thresholds_default_groups():
    out = post.scale_sat_thresholds(BY, scale=0.85)
    assert out["sp_focal"]["kurgany_povrezhdennye"] == pytest.approx(0.3645 * 0.85)
    assert out["sp_or"]["gorodishcha"] == pytest.approx(0.99 * 0.85)
    assert out["ae_or"]["kurgany_tselye"] == 0.7  # untouched
    assert BY["sp_focal"]["kurgany_povrezhdennye"] == 0.3645  # input not mutated


def test_scale_sat_thresholds_rejects_nonpositive():
    with pytest.raises(ValueError):
        post.scale_sat_thresholds(BY, scale=0.0)


def test_sat_thr_relax_by_markup_off_by_default():
    cfg = {"thresholds_by_markup": BY, "sat_thr_relax": {"enabled": False, "scale": 0.5}}
    assert post.sat_thr_relax_by_markup(cfg) == BY


def test_sat_thr_relax_by_markup_on():
    cfg = {
        "thresholds_by_markup": BY,
        "sat_thr_relax": {"enabled": True, "scale": 0.8, "groups": ["sp_focal"]},
    }
    out = post.sat_thr_relax_by_markup(cfg)
    assert out["sp_focal"]["kurgany_povrezhdennye"] == pytest.approx(0.3645 * 0.8)
    assert out["sp_or"]["kurgany_povrezhdennye"] == 0.5  # not in groups


def test_finish_admits_more_when_relax_on():
    """Confidence between default and relaxed threshold → only admitted when enabled."""
    feat = {
        "class_name": "kurgany_povrezhdennye",
        "confidence": 0.33,  # below 0.3645, above 0.3645*0.85≈0.31
        "markup_type": "sp_or",
        "threshold_group": "sp_focal",
        "geom": Point(0, 0).buffer(4.0),
        "region_id": "t",
    }

    cfg_off = {
        "thresholds": {},
        "default_threshold": 0.15,
        "thresholds_by_markup": {"sp_focal": {"kurgany_povrezhdennye": 0.3645}},
        "sat_thr_relax": {"enabled": False, "scale": 0.85},
        "post": {},
    }
    cfg_on = dict(cfg_off)
    cfg_on["sat_thr_relax"] = {"enabled": True, "scale": 0.85}

    off = finish([dict(feat)], cfg_off)
    on = finish([dict(feat)], cfg_on)
    assert len(off) == 0, f"expected rejected when OFF, got {off}"
    assert len(on) == 1, f"expected admitted when ON, got {on}"
    assert on[0]["class_name"] == "kurgany_povrezhdennye"
