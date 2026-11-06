"""Сборка эталона на «трудных» файлах: кириллица в коде, пиксельный профиль,
линии вместо полигонов, явная CRS 3857, дубли между модальностями и конфликт классов."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer

from expds.catalog import read_utm_json, scan_root
from expds.crs import coords_kind, guess_geojson_crs, utm_for_lonlat
from expds.gt import build_gt, render_summary, stats

X0, Y0 = 400_000.0, 6_000_000.0
T = Transformer.from_crs(32636, 3857, always_xy=True)


def _ring(cx, cy, r, n=24, to3857=False):
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    if to3857:
        pts = [T.transform(x, y) for x, y in pts]
    pts = [[float(x), float(y)] for x, y in pts]
    return pts + [pts[0]]


def _fc(geoms, crs=None):
    obj = {"type": "FeatureCollection",
           "features": [{"type": "Feature", "properties": {}, "geometry": g} for g in geoms]}
    if crs:
        obj["crs"] = {"type": "name", "properties": {"name": crs}}
    return obj


def _poly(ring):
    return {"type": "Polygon", "coordinates": [ring]}


def make_tree(root: Path) -> Path:
    s = root / "001_Тест"
    a = s / "06_Тест_разметка"
    a.mkdir(parents=True)
    (s / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    mounds = [(30, 30, 6), (70, 40, 4), (50, 75, 8)]
    (a / "Тест_Li_курганы_целые.geojson").write_text(json.dumps(_fc(
        [_poly(_ring(X0 + x, Y0 + y, r)) for x, y, r in mounds])), encoding="utf-8")
    (a / "Тест_Ае_курганы_целые.geojson").write_text(json.dumps(_fc(           # «Ае» кириллицей
        [_poly(_ring(X0 + x + 1, Y0 + y, r)) for x, y, r in mounds[:2]])), encoding="utf-8")
    x, y, r = mounds[2]
    (a / "Тест_SpOr_курганы_поврежденные.geojson").write_text(json.dumps(_fc(
        [_poly(_ring(X0 + x, Y0 + y, r, to3857=True))], "urn:ogc:def:crs:EPSG::3857")), encoding="utf-8")
    (a / "Тест_EtGr_профиль_14_ObjectPoly.geojson").write_text(json.dumps(_fc(      # пиксели профиля
        [_poly(_ring(120, 15, 5))])), encoding="utf-8")
    (a / "Тест_Li_фортификации.geojson").write_text(json.dumps(_fc(
        [{"type": "LineString", "coordinates": [[X0, Y0], [X0 + 50, Y0 + 5]]}])), encoding="utf-8")
    return root


def test_coords_kind_guards_pixels():
    assert coords_kind([120, 125], [10, 20], "EPSG:32636") == "local"
    assert coords_kind([33.0, 33.1], [55.0, 55.1], "EPSG:32636") == "wgs84"
    assert coords_kind([33.0, 33.1], [55.0, 55.1], "EPSG:32650") == "local"     # далеко от зоны участка
    # Швеция западнее 15° в. д., зона 33 (осевой меридиан 15°). Казахстан — зона 43.
    assert coords_kind([13.1, 13.4], [55.5, 55.7], "EPSG:32633") == "wgs84"
    assert coords_kind([13.1, 13.4], [55.5, 55.7], "EPSG:32636") == "local"
    assert coords_kind([76.2, 76.5], [48.0, 48.2], "EPSG:32643") == "wgs84"


def test_swedish_point_is_implausible_in_a_zone_far_to_the_east():
    from pyproj import Transformer

    from expds.crs import plausible

    lon, lat = 13.2, 55.6
    east = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)
    x, y = east.transform(lon, lat)
    assert plausible("EPSG:32637", (x - 50, y - 50, x + 50, y + 50)) is False
    own = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    x, y = own.transform(lon, lat)
    assert plausible("EPSG:32633", (x - 50, y - 50, x + 50, y + 50)) is True


def test_utm_zone_follows_longitude_not_a_fixed_belt():
    assert utm_for_lonlat(13.2, 55.6) == "EPSG:32633"   # 12–18° в. д.
    assert utm_for_lonlat(11.2, 58.0) == "EPSG:32632"   # западнее 12°
    assert utm_for_lonlat(76.4, 48.1) == "EPSG:32643"   # 72–78° в. д.
    assert utm_for_lonlat(37.6, 55.7) == "EPSG:32637"
    assert utm_for_lonlat(180.0, 60.0) == "EPSG:32660"
    assert utm_for_lonlat(-180.0, 60.0) == "EPSG:32601"
    assert coords_kind([400_000, 400_100], [6e6, 6.0001e6]) == "utm"
    assert guess_geojson_crs(_fc([_poly(_ring(400_000, 6e6, 5))]), None) is None  # UTM без UTM.json


def test_read_utm_json_hemisphere(tmp_path):
    for value, epsg in (("36N", "EPSG:32636"), ("36U", "EPSG:32636"), ("37S", "EPSG:32737"),
                        ("UTM zone 38", "EPSG:32638")):
        p = tmp_path / f"{value.replace(' ', '_')}.json"
        p.write_text(json.dumps({"utm": value}), encoding="utf-8")
        assert read_utm_json(p) == epsg, value
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"crs": "EPSG", "utm": "36N"}), encoding="utf-8")
    assert read_utm_json(bare) == "EPSG:32636"
    only = tmp_path / "only.json"
    only.write_text(json.dumps({"crs": "EPSG"}), encoding="utf-8")
    assert read_utm_json(only) is None
    coded = tmp_path / "coded.json"
    coded.write_text(json.dumps({"crs": "EPSG:32646"}), encoding="utf-8")
    assert read_utm_json(coded) == "EPSG:32646"


def test_gt_diagnostics(tmp_path):
    regions = scan_root(make_tree(tmp_path / "train"))
    report: list[dict] = []
    items, info = build_gt(regions, report=report)
    st = stats(items)
    by = {(r["class_name"], r["markup_type"]): r for r in st["objects"]}
    assert by[("kurgany_tselye", "ae")]["objects"] == 2                        # кириллический файл не потерян
    assert {(p["a"], p["b"]): p["pairs"] for p in st["cross_modality_pairs"]}[("ae", "li")] == 2
    assert {(p["a"], p["b"]): p["pairs"] for p in st["kurgan_class_conflicts"]}[("li", "sp_or")] == 1
    assert ("object_poly", "etgr") not in by                                   # пиксельный профиль не в эталоне
    probs = " ".join(p for r in report for p in r["problems"])
    assert "кириллицей" in probs and "пиксельные" in probs and "линий" in probs
    assert 60 < by[("kurgany_tselye", "li")]["area_m2_p50"] < 130              # площадь в м², а не в единицах 3857
    st.update(info, regions=len(regions))
    share = render_summary(st, report, anonymous=True)
    assert "Тест_Li" not in share and "001_Тест" not in share                  # обезличено


def test_etgr_layers_stay_separate_and_other_layers_merge(tmp_path):
    site = tmp_path / "train" / "001_Тест_Геофизика"
    ann = site / "06_Тест_Геофизика_разметка"
    ann.mkdir(parents=True)
    (site / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    for name, dx in (("EtGr_карта_01_0.5", 0.0), ("EtGr_профиль_01", 0.5),
                     ("MagGr_серый", 0.0), ("MagGr_цвет", 0.5)):
        fc = _fc([_poly(_ring(427_435.0 + dx, 6_070_922.0, 2.0))], crs="urn:ogc:def:crs:EPSG::32636")
        (ann / f"Тест_Геофизика_{name}_ObjectPoly.geojson").write_text(json.dumps(fc), encoding="utf-8")
    items, info = build_gt(scan_root(tmp_path / "train"))
    by = {}
    for it in items:
        by[it["markup_type"]] = by.get(it["markup_type"], 0) + 1
    assert by == {"etgr": 2, "maggr": 1}
    assert info["merges_by_markup_type"].get("etgr", 0) == 0


def test_phase_shift_detects_georeference_offset():
    from expds.lidar import phase_shift
    rng = np.random.default_rng(0)
    a = rng.normal(size=(300, 400)).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(a, 2)
    b = np.roll(np.roll(a, 7, axis=1), -4, axis=0)          # на 7 px восточнее и на 4 px севернее
    dx, dy, peak = phase_shift(a, b)
    assert (dx, dy) == (7.0, -4.0) and peak > 0.5
