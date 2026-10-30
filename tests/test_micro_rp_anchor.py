"""W0-seams: anchor="rp" у post.micro_circles; по умолчанию — побайтно прежнее поведение."""
from shapely.geometry import Polygon, Point
from expds import post

L = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])  # «Г»: центроид вне фигуры


def _f(g, cls="fortifikatsii"):
    return {"geom": g, "class_name": cls, "confidence": 0.5}


def test_default_unchanged():
    a = post.micro_circles([_f(L)], {"radius_m": {"fortifikatsii": 0.05}})
    b = post.micro_circles([_f(L)], {"radius_m": {"fortifikatsii": 0.05}, "anchor": "centroid"})
    assert a[0]["geom"].wkb == b[0]["geom"].wkb


def test_rp_anchor_dominates():
    rp = L.representative_point()
    out = post.micro_circles([_f(L)], {"radius_m": {"fortifikatsii": 0.05}, "anchor": "rp"})
    d = out[0]["geom"]
    assert d.centroid.distance(rp) < 1e-9 and L.contains(d) and L.contains(d.representative_point())


def test_other_classes_untouched():
    g = Point(0, 0).buffer(2)
    out = post.micro_circles([_f(g, "kurgany_tselye")], {"radius_m": {"fortifikatsii": 0.05}, "anchor": "rp"})
    assert out[0]["geom"].wkb == g.wkb
