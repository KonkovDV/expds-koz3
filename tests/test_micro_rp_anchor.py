"""W0-seams: anchor="rp" у post.micro_circles; по умолчанию — побайтно прежнее поведение."""
from shapely.geometry import Polygon, Point
from expds import post

L = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])  # «Г»: центроид вне фигуры


def _f(g, cls="fortifikatsii"):
    return {"geom": g, "class_name": cls, "confidence": 0.5}


def test_centroid_stays_on_the_anchor_the_two_metre_check_uses():
    """Два метра считаются от центроида. Внутренняя точка 32-угольника севернее центра."""
    from shapely.geometry import Point

    centre = Point(4_200_000.0, 7_500_000.0)
    disk = centre.buffer(4.0)
    out = post.micro_circles(
        [_f(disk, "kurgany_tselye")],
        {"radius_m": {"kurgany_tselye": 0.5}},
    )[0]["geom"]
    assert out.centroid.distance(centre) < 1e-6
    shift = out.representative_point().distance(out.centroid)
    assert 0.04 < shift < 0.12


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
