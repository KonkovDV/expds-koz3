"""Кольцо вокруг принятого кургана. В сдаче секции нет."""
import math

from shapely.geometry import Point

from expds.post import rescue_near_kept

LAT = math.radians(55.0)
Y = 6378137.0 * math.log(math.tan(math.pi / 4 + LAT / 2))
COS = math.cos(LAT)
TAU = {"sp_or": {"kurgany_povrezhdennye": 0.3645}}


def _at(dx_ground, conf):
    return {
        "geom": Point(dx_ground / COS, Y).buffer(1.0),
        "class_name": "kurgany_povrezhdennye",
        "confidence": conf,
        "region_name": "a",
        "markup_type": "sp_or",
        "threshold_group": "sp_or",
    }


def test_neighbor_in_the_ring_is_lifted_to_the_threshold():
    feats = [
        _at(0, 0.50),
        _at(40, 0.22),
        _at(5, 0.22),
        _at(200, 0.22),
        _at(40, 0.10),
    ]
    out = rescue_near_kept(feats, {}, TAU)
    conf = [round(f["confidence"], 4) for f in out]
    assert conf[0] == 0.5
    assert conf[1] == 0.3645
    assert conf[2] == 0.22
    assert conf[3] == 0.22
    assert conf[4] == 0.10


def test_another_site_is_not_an_anchor():
    other = _at(40, 0.22)
    other["region_name"] = "b"
    out = rescue_near_kept([_at(0, 0.50), other], {}, TAU)
    assert out[1]["confidence"] == 0.22
