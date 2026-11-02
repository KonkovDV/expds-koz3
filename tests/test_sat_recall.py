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
