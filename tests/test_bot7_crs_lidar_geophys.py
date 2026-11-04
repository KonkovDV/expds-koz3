"""Лидар и геофизика берут CRS участка через единую точку region_crs → catalog.read_utm_json.

UTM.json с географической CRS (как 008_Герите, EPSG:4326) не должен дать ЦМР / дедупликации находок
CRS в градусах: build_dem, DemPrefetch, hillfort.build_dem и geophys.dedupe_points(2,0 м) получают зону UTM.
Плюс страж по исходнику: фазы лидара и геофизики не читают UTM.json/LAS-CRS мимо region_crs.
"""
import inspect
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from conftest import write_las
from expds import geophys, hillfort, pipeline, pix_clean
from expds.catalog import scan_region
from expds.crs import WEB_MERCATOR, to_crs
from expds.lidar import DemPrefetch, build_dem

LON0, LAT0, STEP, N = 44.941, 42.846, 1e-5, 200      # ≈ Герите, зона 38N


def _region(tmp_path):
    reg = tmp_path / "901_Синтетика"
    tif = reg / "05_Синтетика_Or_ортофото" / "01_Синтетика_Or_ортофото.tif"
    tif.parent.mkdir(parents=True)
    with rasterio.open(tif, "w", driver="GTiff", height=N, width=N, count=3, dtype="uint8",
                       transform=from_origin(LON0, LAT0, STEP, STEP), crs="EPSG:4326") as ds:
        ds.write(np.full((3, N, N), 90, np.uint8))
    write_las(reg / "01_Синтетика_La_точки" / "01_Синтетика_La_точки.las", n=20_000, wkt=False)
    (reg / "UTM.json").write_text(json.dumps({"utm": "WGS84", "crs": "urn:ogc:def:crs:EPSG::4326"}),
                                  encoding="utf-8")
    return scan_region(reg)


def test_lidar_dem_gets_projected_region_crs(tmp_path):
    r = _region(tmp_path)
    assert r.las, "LAS не найден в синтетическом участке"
    for cfg in (None, {"crs": {"policy": "utm_json"}}, {"crs": {"policy": "vote"}}):
        assert pipeline.region_crs(r, cfg) == "EPSG:32638"
    crs = pipeline.region_crs(r, {"crs": {"policy": "vote"}})
    las = r.las[0]
    assert build_dem(las, res=0.5, fill="delaunay", crs=crs).crs == "EPSG:32638"
    pf = DemPrefetch([(las, crs)], 1)
    try:
        assert pf.get(las, 0.5, "delaunay", crs).crs == "EPSG:32638"
    finally:
        pf.close()
    assert hillfort.build_dem(las, crs, hillfort.params_from({})).crs == "EPSG:32638"


def test_geophys_dedupe_2m_in_projected_region_crs(tmp_path):
    r = _region(tmp_path)
    crs = pipeline.region_crs(r, {})
    assert crs == "EPSG:32638"
    x0, y0 = 300000.0, 4746000.0
    pts = [to_crs(Point(x0, y0), crs, WEB_MERCATOR), to_crs(Point(x0 + 1.5, y0), crs, WEB_MERCATOR),
           to_crs(Point(x0 + 5.0, y0), crs, WEB_MERCATOR)]
    feats = [{"geom": g.buffer(0.5), "class_name": "finds_points", "confidence": c, "source": s}
             for g, c, s in zip(pts, (0.9, 0.8, 0.7), ("gray", "color", "color"))]
    out = geophys.dedupe_points(feats, 2.0, crs)
    assert [f["confidence"] for f in out] == [0.9, 0.7]          # 1,5 м — дубль, 5 м — отдельная находка


def test_lidar_and_geophys_phases_use_single_crs_point():
    for fn in (pipeline.geophys_feats, pipeline.lidar_feats, pipeline.li_dev_feats, pipeline.geo_model_feats,
               pix_clean.region_feats):
        src = inspect.getsource(fn)
        assert "region_crs(reg, cfg)" in src, fn.__name__
        for bad in ("read_utm_json", "_read_utm_json_raw", "las_crs(", "utm_json"):
            assert bad not in src, (fn.__name__, bad)
    src = inspect.getsource(pipeline.region_crs)
    assert "read_utm_json(" in src
    run_src = inspect.getsource(pipeline)
    assert "DemPrefetch([(las, region_crs(r, cfg))" in run_src
