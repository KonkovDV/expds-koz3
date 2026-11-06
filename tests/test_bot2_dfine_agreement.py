""" A3/A4: agreement_m — D-FINE только рядом с U-Net sp_or (default OFF)."""
from __future__ import annotations

from shapely.geometry import Point

from expds import sp_dfine


def _c(x, y, r=4.0, **props):
    return dict(geom=Point(x, y).buffer(r), **props)


def test_agreement_m_default_off():
    st = sp_dfine.settings({})
    assert st["agreement_m"] == 0.0
    st2 = sp_dfine.settings({"sp_dfine": {"agreement_m": 50}})
    assert st2["agreement_m"] == 50


def test_filter_agreement_off_passthrough():
    unet = _c(1e6, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
              threshold_group="sp_or", region_name="r1", confidence=0.9)
    far = _c(1e6 + 400, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
             threshold_group="sp_dfine", region_name="r1", confidence=0.65)
    assert sp_dfine.filter_agreement([far], [unet], 0) == [far]
    assert sp_dfine.filter_agreement([far], [unet], -1) == [far]


def test_filter_agreement_keeps_near_drops_far():
    # ~ WebMercator metres ≈ ground at mid-latitudes after ground_scale; use large Δ for far.
    unet = _c(1e6, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
              threshold_group="sp_or", region_name="r1", confidence=0.9)
    near = _c(1e6 + 5, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
              threshold_group="sp_dfine", region_name="r1", confidence=0.65)
    far = _c(1e6 + 500, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
             threshold_group="sp_dfine", region_name="r1", confidence=0.65)
    out = sp_dfine.filter_agreement([near, far], [unet], 50.0)
    assert len(out) == 1
    assert out[0] is near


def test_filter_agreement_ignores_dfine_anchors():
    """Якоря — только не-sp_dfine; другой D-FINE не спасает далёкий FP."""
    other_dfine = _c(1e6, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
                     threshold_group="sp_dfine", region_name="r1", confidence=0.7)
    far = _c(1e6 + 10, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
             threshold_group="sp_dfine", region_name="r1", confidence=0.65)
    assert sp_dfine.filter_agreement([far], [other_dfine], 50.0) == []


def test_filter_agreement_empty_anchors_drops_all():
    far = _c(1e6, 7e6, class_name="kurgany_povrezhdennye", markup_type="sp_or",
             threshold_group="sp_dfine", region_name="r1", confidence=0.65)
    assert sp_dfine.filter_agreement([far], [], 50.0) == []
