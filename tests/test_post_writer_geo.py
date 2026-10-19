import json

import numpy as np
from shapely.geometry import MultiPolygon, Point, box

from conftest import UTM, write_mag_tif
from expds import geophys, post, writer
from expds.crs import WEB_MERCATOR, to_crs

BASE = to_crs(Point(500_000, 6_100_000), UTM, WEB_MERCATOR)


def feat(geom, cls="kurgany_tselye", mt="ae", src="a.tif", conf=0.5, region="r1"):
    return {"geom": geom, "class_name": cls, "markup_type": mt, "source": src, "confidence": conf,
            "region_name": region}


def sq(dx, dy, s=5):
    return box(BASE.x + dx, BASE.y + dy, BASE.x + dx + s, BASE.y + dy + s)


def test_merge_only_across_images_within_markup():
    feats = [feat(sq(0, 0), src="a.tif", conf=0.3), feat(sq(2, 2), src="b.tif", conf=0.7),   # слить
             feat(sq(20, 0), src="a.tif"), feat(sq(22, 2), src="a.tif"),                     # один снимок
             feat(sq(40, 0), mt="li"), feat(sq(40, 0), mt="ae", src="c.tif")]                 # разные типы
    out = post.merge_within_markup(feats)
    assert len(out) == 5
    merged = [f for f in out if f["geom"].area > 30 and f["markup_type"] == "ae"]
    assert len(merged) == 1 and merged[0]["confidence"] == 0.7


def test_finalize_splits_multipolygon_and_caps_vertices():
    mp = MultiPolygon([sq(0, 0), sq(50, 50)])
    big = BASE.buffer(30, quad_segs=2000)        # ~8000 вершин
    out = post.finalize([feat(mp), feat(big)])
    assert len(out) == 3 and all(f["geom"].geom_type == "Polygon" for f in out)
    assert max(len(f["geom"].exterior.coords) for f in out) < 3000


def test_writer_roundtrip_and_validation(tmp_path):
    p = writer.write_geojson(post.finalize([feat(sq(0, 0)), feat(sq(10, 0), cls="finds_points")]),
                             tmp_path / "r.geojson")
    rep = writer.validate(p)
    assert rep["ok"], rep
    bad = json.loads(p.read_text(encoding="utf-8"))
    bad["features"][0]["properties"]["class_name"] = "kurgan"
    bad["features"][1]["geometry"] = {"type": "Point", "coordinates": [BASE.x, BASE.y]}
    rep = writer.validate(bad)
    assert not rep["ok"] and len(rep["errors"]) == 2


def test_thresholds_and_cap():
    feats = [feat(sq(i * 10, 0), conf=c) for i, c in enumerate([0.05, 0.2, 0.9])]
    assert len(post.apply_thresholds(feats, {"kurgany_tselye": 0.15})) == 2
    assert [f["confidence"] for f in post.cap_features(feats, 1)] == [0.9]


def test_geophys_points_and_polys(tmp_path):
    path, pts = write_mag_tif(tmp_path / "m.tif")
    found = geophys.finds_points(path, "r", "maggr")
    assert found and all(f["class_name"] == "finds_points" for f in found)
    cents = [to_crs(f["geom"], WEB_MERCATOR, UTM).centroid for f in found]
    hits = sum(min(c.distance(Point(x, y)) for c in cents) <= 2.0 for x, y in pts)
    assert hits == len(pts)
    polys = geophys.object_polys(path, "r", "maggr")
    areas = [to_crs(f["geom"], WEB_MERCATOR, UTM).area for f in polys]
    assert any(2.0 < a < 10.0 for a in areas)
    assert np.isfinite([f["confidence"] for f in polys]).all()
