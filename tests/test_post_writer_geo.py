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


def test_geophys_rasters_params_and_masks():
    from pathlib import Path

    from shapely.geometry import box

    files = [Path(n) for n in ("01_X_MagGr_серый.tif", "02_X_MagGr_цвет.tif")]
    assert [f.name for f in geophys.select_rasters(files, "maggr")] == ["01_X_MagGr_серый.tif"]
    et = [Path(n) for n in ("01_X_EtGr_карта_04_0.75.tif", "02_X_EtGr_карта_09_2.5.tif", "03_X_EtGr_профиль_01.tif")]
    assert [f.name for f in geophys.select_rasters(et, "etgr")] == ["01_X_EtGr_карта_04_0.75.tif",
                                                                    "03_X_EtGr_профиль_01.tif"]
    cfg = {"background_m": 4.0, "maggr": {"object_poly": {"poly_k": 1.5, "use_noise": True, "min_confidence": 0.2,
                                                          "confidence": "rank"}}}
    p, o = geophys.params_for(cfg, "maggr", "object_poly")
    assert p.poly_k == 1.5 and p.background_m == 4.0 and (p.min_area_m2, p.max_area_m2) == geophys.AREA_M2["maggr"]
    assert o == {"use_noise": True, "min_confidence": 0.2, "rank": True}
    p2, o2 = geophys.params_for(cfg, "etgr", "object_poly")
    assert p2.poly_k == geophys.GeoParams().poly_k and o2["use_noise"] is False and o2["rank"] is False
    ranked = geophys.rank_within_source(
        [{"region_name": "r", "source": s, "confidence": c} for s, c in (("a", 5.0), ("a", 1.0), ("b", 0.1))])
    assert [f["confidence"] for f in ranked] == [1.0, 0.5, 1.0]
    feats = [{"geom": box(x, 0, x + 1, 1)} for x in (0, 5, 20)]
    kept = geophys.apply_masks(feats, boundary=box(-1, -1, 10, 2), noise=box(4, -1, 7, 2))
    assert [f["geom"].bounds[0] for f in kept] == [0]


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
