from shapely.geometry import Polygon
from expds import post, watchdog

L = Polygon([(0, 0), (10, 0), (10, 10), (8, 10), (8, 2), (0, 2)])  # L-образный


def _f(cls, mk):
    return {"class_name": cls, "markup_type": mk, "threshold_group": mk, "confidence": 0.9, "region_name": "r", "geom": L}


def test_hull_only_etgr_terfl_object_poly():
    out = post.convex_hull_object_poly([_f("object_poly", "etgr"), _f("object_poly", "terfl"),
                                        _f("object_poly", "maggr"), _f("finds_points", "etgr")])
    assert [round(f["geom"].area, 6) for f in out] == [round(L.convex_hull.area, 6)] * 2 + [round(L.area, 6)] * 2


def test_flag_default_off_and_on():
    cfg = {"thresholds": {}, "default_threshold": 0.0}
    off = watchdog.finish([_f("object_poly", "etgr")], cfg)
    on = watchdog.finish([_f("object_poly", "etgr")], dict(cfg, post={"geo_convex_hull": True}))
    assert abs(off[0]["geom"].area - L.area) < 1e-9
    assert abs(on[0]["geom"].area - L.convex_hull.area) < 1e-9
