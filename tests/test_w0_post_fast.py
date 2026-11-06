"""w0: векторный post.finalize даёт те же объекты, что finalize_ref (поэлементный эталон)."""
import math
import random

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from expds import post


def _feats():
    rnd = random.Random(7)
    out = []
    for i in range(300):
        x, y = rnd.uniform(0, 1e4), rnd.uniform(0, 1e4)
        k = i % 6
        if k == 0:
            g = Point(x, y).buffer(rnd.uniform(0.01, 20), quad_segs=rnd.choice([4, 64, 400]))
        elif k == 1:
            g = Polygon([(x, y), (x + 5, y + 5), (x + 5, y), (x, y + 5)])        # бабочка — невалидный
        elif k == 2:
            g = MultiPolygon([Point(x, y).buffer(3), Point(x + 10, y).buffer(0.01)])
        elif k == 3:
            g = Polygon()
        elif k == 4:
            ring = [(x + 30 * math.cos(t / 300 * 2 * math.pi) * (1 + 0.1 * rnd.random()),
                     y + 30 * math.sin(t / 300 * 2 * math.pi)) for t in range(300)]
            g = Polygon(ring, [list(Point(x, y).buffer(2, quad_segs=200).exterior.coords)])
        else:
            g = GeometryCollection([Point(x, y).buffer(1), LineString([(x, y), (x + 1, y)])])
        out.append({"geom": g, "class_name": "kurgany_tselye", "confidence": rnd.random(), "i": i})
    return out


def test_finalize_matches_reference():
    a, b = post.finalize(_feats()), post.finalize_ref(_feats())
    assert len(a) == len(b) and len(a) > 100
    for x, y in zip(a, b):
        assert x["i"] == y["i"] and x["geom"].equals_exact(y["geom"], 0.0)
        assert post._vertex_count(x["geom"]) < post.MAX_VERTICES


def test_merge_within_markup_matches_reference(monkeypatch):
    rnd = random.Random(3)
    feats = []
    for i in range(400):
        x, y = rnd.uniform(0, 300), rnd.uniform(0, 300)
        feats.append({"geom": Point(x, y).buffer(rnd.uniform(1, 8)), "class_name": rnd.choice(["kurgany_tselye", "arkhitektury"]),
                      "confidence": round(rnd.random(), 2), "region_name": "r", "markup_type": rnd.choice(["li", "ae"]),
                      "source": rnd.choice("abc"), "i": i})
    a = post.merge_within_markup(feats)
    monkeypatch.setenv("EXPDS_POST_FAST", "0")
    b = post.merge_within_markup(feats)
    assert len(a) == len(b) and len(a) < len(feats)
    for x, y in zip(a, b):
        assert x["i"] == y["i"] and x["geom"].equals_exact(y["geom"], 0.0)


def test_micro_circles_matches_reference(monkeypatch):
    spec = {"radius_m": {"kurgany_tselye": 0.5, "arkhitektury": 0.25}}
    feats = _feats()
    for j, f in enumerate(feats):
        if j % 3 == 0:
            f["class_name"] = "arkhitektury"
        f["geom"] = f["geom"] if f["geom"].is_valid else f["geom"].buffer(0)
    a = post.micro_circles(feats, spec)
    monkeypatch.setenv("EXPDS_POST_FAST", "0")
    b = post.micro_circles(feats, spec)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["i"] == y["i"] and x["geom"].equals_exact(y["geom"], 0.0)


def _kurgan(geom, source, confidence, cls="kurgany_tselye"):
    return {"geom": geom, "class_name": cls, "markup_type": "li", "region_name": "r",
            "source": source, "confidence": confidence, "threshold_group": "li"}


def test_failed_union_keeps_that_group_and_the_other_class(monkeypatch):
    """Одно слияние не роняет цепочку: эта группа остаётся по отдельности, соседний класс на месте."""
    a = _kurgan(box(0, 0, 2, 2), "a", 0.9)
    b = _kurgan(box(1, 1, 3, 3), "b", 0.4)
    other = _kurgan(box(100, 100, 102, 102), "a", 0.8, "gorodishcha")
    merged = post.merge_within_markup([a, b, other])
    assert sum(f["class_name"] == "kurgany_tselye" for f in merged) == 1
    monkeypatch.setattr(post, "unary_union", lambda _geoms: (_ for _ in ()).throw(RuntimeError("union")))
    kept = post.merge_within_markup([a, b, other])
    assert sum(f["class_name"] == "kurgany_tselye" for f in kept) == 2
    assert sum(f["class_name"] == "gorodishcha" for f in kept) == 1
