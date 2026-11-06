"""UTM.json с географической CRS (как 008_Герите: EPSG:4326) → зона UTM по центру участка.

До фикса read_utm_json отдавал EPSG:4326 как есть: sp_dfine.Scene строилась с шагом 0,5°
(1×1 px, ноль окон), gor_gate/gor_dense получали EPSG 4326, li строил ЦМР с CRS в градусах.
"""
import json
import logging

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from expds import gor_gate, sp_dfine
from expds.catalog import read_utm_json, scan_region
from expds.crs import WEB_MERCATOR, project_region_crs, to_crs
from expds.pipeline import region_crs

LON0, LAT0, STEP, N = 44.941, 42.846, 1e-5, 300      # ≈ Герите: зона 38N, ≈245×333 м


def _region(tmp_path, utm_json, crs="EPSG:4326", south=False):
    reg = tmp_path / "900_Синтетика"
    tif = reg / "05_Синтетика_SpOr_спутник" / "01_Синтетика_SpOr_спутник.tif"
    tif.parent.mkdir(parents=True)
    lat0 = -LAT0 if south else LAT0
    with rasterio.open(tif, "w", driver="GTiff", height=N, width=N, count=4, dtype="uint8",
                       transform=from_origin(LON0, lat0, STEP, STEP), crs=crs) as ds:
        ds.write(np.random.default_rng(0).integers(1, 255, (4, N, N), dtype=np.uint8))
    (reg / "UTM.json").write_text(json.dumps(utm_json), encoding="utf-8")
    return reg, tif


def test_geographic_utm_json_becomes_center_zone(tmp_path, caplog):
    reg, tif = _region(tmp_path, {"utm": "WGS84", "crs": "urn:ogc:def:crs:EPSG::4326"})
    with caplog.at_level(logging.WARNING):
        crs = read_utm_json(reg / "UTM.json")
    assert crs == "EPSG:32638"
    assert any("географическая" in r.getMessage() and "EPSG:32638" in r.getMessage() for r in caplog.records)
    r = scan_region(reg)
    assert region_crs(r) == "EPSG:32638"                      # li / geophys / optics fallback
    assert region_crs(r, {"crs": {"policy": "vote"}}) == "EPSG:32638"
    assert gor_gate._epsg(r) == 32638                          # gor_gate и gor_dense

    # sp_dfine: путь пакета crs_for=read_utm_json → сцена в метрах, а не 1×1 px
    utm = sp_dfine.utm_for(tif, read_utm_json(r.utm_json))
    sc = sp_dfine.Scene(tif, utm, 0.5)
    assert sc.W > 400 and sc.H > 600, (sc.W, sc.H)
    assert abs(sc.T.a - 0.5) < 1e-9
    # запись в 3857: центр сцены совпадает с центром растра
    from shapely.geometry import Point
    cx, cy = sc.T * (sc.W / 2, sc.H / 2)
    a = to_crs(Point(cx, cy), utm, WEB_MERCATOR)
    b = to_crs(Point(LON0 + STEP * N / 2, LAT0 - STEP * N / 2), "EPSG:4326", WEB_MERCATOR)
    assert a.distance(b) < 2.0


def test_utm_for_guard_on_geographic_fallback(tmp_path):
    _reg, tif = _region(tmp_path, {"crs": "EPSG:4326"})
    assert sp_dfine.utm_for(tif, "EPSG:4326") == "EPSG:32638"
    assert sp_dfine.utm_for(tif, "EPSG:32636") == "EPSG:32636"   # проекционная — как есть


def test_south_and_projected_unchanged(tmp_path):
    reg, _ = _region(tmp_path, {"crs": "EPSG:4326"}, south=True)
    assert read_utm_json(reg / "UTM.json") == "EPSG:32738"
    (reg / "UTM.json").write_text(json.dumps({"utm": "36N", "crs": "urn:ogc:def:crs:EPSG::32636"}), encoding="utf-8")
    assert read_utm_json(reg / "UTM.json") == "EPSG:32636"


def test_geographic_without_georeferenced_files_kept(tmp_path, caplog):
    reg = tmp_path / "901_Пусто"
    reg.mkdir()
    with caplog.at_level(logging.WARNING):
        assert project_region_crs("EPSG:4326", reg) == "EPSG:4326"
    assert any("центр участка не определить" in r.getMessage() for r in caplog.records)
    assert project_region_crs(None, reg) is None
