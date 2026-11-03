"""bot-7: [sat_band_verify] / [sat_agree_accept] / [sat_cross_nms] default OFF."""
from __future__ import annotations

import pytest
from shapely.geometry import Point

from expds import sat_recall
from expds.watchdog import finish


def _kurgan(conf, x, y, group="sp_focal", markup="sp_or", region="r1",
            cls="kurgany_povrezhdennye", verify_p=None, **extra):
    f = {
        "class_name": cls,
        "confidence": conf,
        "markup_type": markup,
        "threshold_group": group,
        "geom": Point(x, y).buffer(4.0),
        "region_name": region,
        "region_id": region,
    }
    if verify_p is not None:
        f["verify_p"] = verify_p
    f.update(extra)
    return f


BY = {"sp_focal": {"kurgany_povrezhdennye": 0.40},
      "sp_focal2": {"kurgany_povrezhdennye": 0.50}}


def _cfg(**flags):
    base = {
        "thresholds": {},
        "default_threshold": 0.15,
        "thresholds_by_markup": BY,
        "post": {},
        "sat_band_verify": {"enabled": False},
        "sat_agree_accept": {"enabled": False},
        "sat_cross_nms": {"enabled": False},
    }
    base.update(flags)
    return base


def test_any_enabled_false_by_default():
    assert sat_recall.any_enabled({}) is False
    assert sat_recall.any_enabled(_cfg()) is False


def test_band_verify_off_rejects_midband():
    feat = _kurgan(0.37, 0, 0, verify_p=0.9)  # below 0.40, above 0.36
    off = finish([dict(feat)], _cfg())
    assert len(off) == 0


def test_band_verify_admits_with_verify():
    feat = _kurgan(0.37, 0, 0, verify_p=0.55)
    cfg = _cfg(sat_band_verify={
        "enabled": True, "scale": 0.90, "verify_floor": 0.40,
        "groups": ["sp_focal"], "classes": ["kurgany_povrezhdennye"],
    })
    on = finish([dict(feat)], cfg)
    assert len(on) == 1
    assert on[0].get("sat_band") is True


def test_band_verify_rejects_low_verify():
    feat = _kurgan(0.37, 0, 0, verify_p=0.10)
    cfg = _cfg(sat_band_verify={
        "enabled": True, "scale": 0.90, "verify_floor": 0.40,
        "groups": ["sp_focal"],
    })
    assert finish([dict(feat)], cfg) == []


def test_band_verify_rejects_no_verify_by_default():
    feat = _kurgan(0.37, 0, 0)  # no verify_p
    cfg = _cfg(sat_band_verify={
        "enabled": True, "scale": 0.90, "verify_floor": 0.40,
        "groups": ["sp_focal"],
    })
    assert finish([dict(feat)], cfg) == []


def test_band_keeps_above_orig_thr_without_verify():
    feat = _kurgan(0.45, 0, 0)  # above 0.40
    cfg = _cfg(sat_band_verify={
        "enabled": True, "scale": 0.90, "verify_floor": 0.99,
        "groups": ["sp_focal"],
    })
    assert len(finish([dict(feat)], cfg)) == 1


def test_agree_rejects_radius_8():
    with pytest.raises(ValueError, match="8"):
        sat_recall.mark_agree(
            [_kurgan(0.3, 0, 0)],
            {"enabled": True, "radius_m": 8.0, "groups": ["sp_focal", "sp_focal2"],
             "classes": ["kurgany_povrezhdennye"], "scale": 0.85},
        )


def test_agree_accept_admits_paired_below_thr():
    # Two groups, ~10 m apart in 3857 near equator scale≈1 → within 16 m
    a = _kurgan(0.35, 0, 0, group="sp_focal")       # thr 0.40
    b = _kurgan(0.55, 10, 0, group="sp_focal2", markup="sp_focal2")  # thr 0.50, above
    cfg = _cfg(sat_agree_accept={
        "enabled": True, "radius_m": 16.0, "scale": 0.85,
        "groups": ["sp_focal", "sp_focal2"],
        "classes": ["kurgany_povrezhdennye"],
    })
    out = finish([dict(a), dict(b)], cfg)
    classes_conf = {(f["threshold_group"], round(f["confidence"], 2)) for f in out}
    assert ("sp_focal", 0.35) in classes_conf  # admitted via agree
    assert ("sp_focal2", 0.55) in classes_conf


def test_agree_alone_below_thr_still_rejected():
    a = _kurgan(0.35, 0, 0, group="sp_focal")
    cfg = _cfg(sat_agree_accept={
        "enabled": True, "radius_m": 16.0, "scale": 0.85,
        "groups": ["sp_focal", "sp_focal2"],
    })
    assert finish([dict(a)], cfg) == []


def test_cross_nms_keeps_higher_conf():
    a = _kurgan(0.6, 0, 0, group="sp_focal")
    b = _kurgan(0.9, 1, 0, group="sp_focal2", markup="sp_focal2")  # ~1 m away
    cfg = _cfg(sat_cross_nms={
        "enabled": True, "nms_m": 4.0,
        "groups": ["sp_focal", "sp_focal2"],
        "classes": ["kurgany_povrezhdennye"],
    })
    # both above thr
    out = finish([dict(a), dict(b)], cfg)
    assert len(out) == 1
    assert out[0]["confidence"] == 0.9


def test_finish_identity_when_all_off():
    """Flags OFF → same admit/reject as plain apply_thresholds."""
    low = _kurgan(0.35, 0, 0)
    high = _kurgan(0.45, 100, 0)
    cfg = _cfg()
    out = finish([dict(low), dict(high)], cfg)
    assert len(out) == 1
    assert out[0]["confidence"] == 0.45
