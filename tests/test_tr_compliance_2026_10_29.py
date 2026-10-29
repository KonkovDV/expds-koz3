"""Сверка с утверждённым ТР КОЗ №3 (протокол №14 от 22.10.2026) и разъяснениями 29.10.

Прил. 1 п. 4: на валидации лидар — только LAS, без Li-отмывок. Прил. 3: лимиты файла
и замечания валидатора геометрии (> 50 — отказ). См. docs/COMPLIANCE_2026-10-29.md.
"""
import json
import math
from pathlib import Path

from shapely.geometry import Point, Polygon, box

from expds import catalog, pipeline, post, writer


def _f(geom, cls="kurgany_tselye", conf=0.5, mt="li"):
    return {"geom": geom, "class_name": cls, "confidence": conf, "region_name": "region_0001", "markup_type": mt}


X0, Y0 = 4_000_000.0, 7_000_000.0


# --- Прил. 1 п. 4: LAS без Li ------------------------------------------------------------

def test_region_with_las_only_has_li_type(tmp_path):
    reg_dir = tmp_path / "region_0001"
    reg_dir.mkdir()
    (reg_dir / "region_0001_cloud.las").write_bytes(b"not a real las")
    reg = catalog.scan_region(reg_dir)
    assert [p.name for p in reg.las] == ["region_0001_cloud.las"]
    assert reg.li_rasters == []
    assert "li" in reg.markup_types


def test_li_tif_never_reaches_inference_rasters(tmp_path):
    """Li-отмывка (трейн) кладётся в li_rasters и не попадает в ветки снимков."""
    reg_dir = tmp_path / "region_0002"
    reg_dir.mkdir()
    (reg_dir / "Shumgora_Li.tif").write_bytes(b"")
    reg = catalog.scan_region(reg_dir)
    assert [p.name for p in reg.li_rasters] == ["Shumgora_Li.tif"]
    assert "li" not in reg.rasters
    assert all("Li" not in p.name for files in reg.rasters.values() for p in files)


class _FakeDem:
    crs = "EPSG:32637"
    info: dict = {}


def test_lidar_branch_runs_from_las_without_li(tmp_path, monkeypatch):
    """Без моделей и без Li-тифов LAS всё равно даёт лидарные объекты (запасной детектор)."""
    reg_dir = tmp_path / "region_0003"
    reg_dir.mkdir()
    (reg_dir / "region_0003.las").write_bytes(b"x")
    reg = catalog.scan_region(reg_dir)
    from expds import lidar

    calls = []
    monkeypatch.setattr(pipeline, "_cached_segmenters", lambda *a, **k: [])
    monkeypatch.setattr(lidar, "build_dem", lambda las, **k: calls.append(las.name) or _FakeDem())

    def fake_cpu(d, region, source, cfg, root, cand_p):
        return [_f(Point(X0, Y0).buffer(3.0)) | {"source": source}]

    monkeypatch.setattr(pipeline, "cpu_kurgans", fake_cpu)
    guard = pipeline.TimeGuard(1500)
    out = pipeline.lidar_feats(reg, {}, guard, {}, tmp_path)
    assert calls == ["region_0003.las"]
    assert len(out) == 1 and out[0]["markup_type"] == "li" and out[0]["source"] == "region_0003.las"


# --- Прил. 3: замечания валидатора, точность координат, размер --------------------------

def test_rounding_keeps_thin_polygon_valid():
    thin = Polygon([(X0, Y0), (X0 + 1.0, Y0), (X0 + 0.5, Y0 + 0.003)])   # после round(·, 2) площадь 0
    fc = writer.to_feature_collection([_f(thin)])
    assert len(fc["features"]) == 1
    ring = fc["features"][0]["geometry"]["coordinates"][0]
    assert Polygon(ring).is_valid and Polygon(ring).area > 0
    assert writer.tr_geometry_remarks(fc) == []


def test_normal_polygons_keep_two_decimals():
    fc = writer.to_feature_collection([_f(Point(X0 + 0.123456, Y0 + 0.987654).buffer(0.87, quad_segs=8))])
    for x, y in fc["features"][0]["geometry"]["coordinates"][0]:
        assert round(x, 2) == x and round(y, 2) == y


def test_invalid_geometry_dropped_not_written():
    bow = Polygon([(X0, Y0), (X0 + 10, Y0 + 10), (X0 + 10, Y0), (X0, Y0 + 10)])  # самопересечение
    assert not bow.is_valid
    fc = writer.to_feature_collection([_f(bow), _f(box(X0, Y0, X0 + 5, Y0 + 5))])
    assert len(fc["features"]) == 1


def test_tr_remarks_detected_and_limit():
    bow = {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 10, Y0 + 10], [X0 + 10, Y0], [X0, Y0 + 10], [X0, Y0]]]}
    zero = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    unclosed = {"type": "Polygon", "coordinates": [[[X0, Y0], [X0 + 1, Y0], [X0 + 1, Y0 + 1], [X0, Y0 + 1]]]}
    feats = [{"type": "Feature", "geometry": g, "properties": {"class_name": "object_poly"}}
             for g in (bow, zero, unclosed)]
    rem = writer.tr_geometry_remarks({"type": "FeatureCollection", "features": feats},
                                     raw_text='{"x": 4000000.1234567890123456}')
    text = " ".join(rem)
    assert "15 знаками" in text and "самопересечение" in text and "(0, 0)" in text and "незамкнутое" in text
    many = {"type": "FeatureCollection", "features": [dict(feats[0]) for _ in range(26)]}   # 2 замечания на объект (самопересечение + площадь 0)
    rep = writer.validate(many)
    assert not rep["ok"] and any("Прил. 3" in e for e in rep["errors"])
    rep_ok = writer.validate({"type": "FeatureCollection", "features": [dict(feats[0]) for _ in range(25)]})
    assert rep_ok["ok"]


def test_written_file_has_no_remarks(tmp_path):
    feats = [_f(Point(X0 + 7 * i, Y0 + 3 * i).buffer(0.05 * 1.75, quad_segs=8), cls="object_poly", mt="etgr")
             for i in range(300)]
    p = writer.write_geojson(feats, tmp_path / "result.geojson")
    rep = writer.validate(p)
    assert rep["ok"] and rep["summary"]["tr_remarks"] == 0 and rep["summary"]["features"] == 300


def test_size_guard_noop_for_normal_output():
    feats = [_f(Point(X0 + 10 * i, Y0).buffer(1.0)) for i in range(10)]
    assert writer._fit_size(feats) == writer._dumps(feats)


def test_size_guard_trims_by_class_quota():
    feats = ([_f(Point(X0 + 10 * i, Y0).buffer(1.0), cls="kurgany_povrezhdennye", conf=0.01 * i) for i in range(60)]
             + [_f(Point(X0 + 10 * i, Y0 + 100).buffer(1.0), cls="fortifikatsii", conf=0.9) for i in range(60)])
    full = len(writer._dumps(feats).encode())
    budget = full // 2
    text = writer._fit_size(feats, budget)
    assert len(text.encode()) <= budget
    got = json.loads(text)["features"]
    by = {}
    for f in got:
        by[f["properties"]["class_name"]] = by.get(f["properties"]["class_name"], 0) + 1
    assert by["kurgany_povrezhdennye"] > by.get("fortifikatsii", 0)         # квоты по весам метрики (30 vs 5)
    confs = sorted(f["properties"]["confidence"] for f in got if f["properties"]["class_name"] == "kurgany_povrezhdennye")
    assert min(confs) >= 0.01 * (60 - len(confs)) - 1e-9                     # внутри класса — самые уверенные


# --- Прил. 1/разъяснения: дедупликация в пределах типа, EtGr по снимкам ------------------

def test_merge_within_markup_does_not_cross_types_and_keeps_etgr():
    g = Point(X0, Y0).buffer(5.0)
    feats = [_f(g, mt="li") | {"source": "a.las"}, _f(g.buffer(0.5), mt="ae") | {"source": "a.tif"},
             _f(g, cls="object_poly", mt="etgr") | {"source": "map1.tif"},
             _f(g.buffer(0.5), cls="object_poly", mt="etgr") | {"source": "map2.tif"},
             _f(g, mt="sp_or") | {"source": "s1.tif"}, _f(g.buffer(0.3), mt="sp_or") | {"source": "s2.tif"}]
    out = post.merge_within_markup(feats)
    by = {}
    for f in out:
        by[f["markup_type"]] = by.get(f["markup_type"], 0) + 1
    assert by == {"li": 1, "ae": 1, "etgr": 2, "sp_or": 1}


def test_fast_writer_path_also_keeps_thin_polygon_valid(monkeypatch):
    """Флаг [speed] fast_writer (харвест 29.10) не обходит проверку валидности после округления."""
    monkeypatch.setenv("EXPDS_SPEED_FAST_WRITER", "1")
    thin = Polygon([(X0, Y0), (X0 + 1.0, Y0), (X0 + 0.5, Y0 + 0.003)])
    normal = Polygon([(X0, Y0), (X0 + 10.0, Y0), (X0 + 10.0, Y0 + 10.0), (X0, Y0 + 10.0)])
    fast = writer.to_feature_collection([_f(thin), _f(normal)])
    monkeypatch.setenv("EXPDS_SPEED_FAST_WRITER", "0")
    slow = writer.to_feature_collection([_f(thin), _f(normal)])
    assert fast == slow
    assert writer.tr_geometry_remarks(fast) == []
