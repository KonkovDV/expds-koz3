""" crsfix (находка): UTM.json с географической CRS (008_Герите: urn:ogc:def:crs:EPSG::4326) не должна
становиться CRS сцены D-FINE / шлюза городищ — иначе шаг 0,5° вместо 0,5 м, сцена 1×1 px, 0 предсказаний."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from expds import dfine_aeor as A
from expds import gor_gate, sp_dfine
from expds.catalog import read_utm_json


def _geo_tif(p: Path, n: int = 512, deg: float = 4e-6) -> Path:
    import rasterio
    from rasterio.transform import from_origin
    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(45.1, 57.0, deg, deg)) as d:
        d.write(np.full((3, n, n), 80, dtype=np.uint8))
    return p


def _utm_json(p: Path, crs: str) -> Path:
    p.write_text(json.dumps({"utm": "WGS84", "crs": crs}), encoding="utf-8")
    return p


def test_geographic_utm_json_falls_back_to_raster_utm_zone(tmp_path):
    tif = _geo_tif(tmp_path / "r" / "05_r_Or_ортофото" / "01_r_Or_ортофото.tif")
    fb = read_utm_json(_utm_json(tmp_path / "r" / "UTM.json", "urn:ogc:def:crs:EPSG::4326"))
    # v4 (слияние с v3fix): read_utm_json — единая точка, градусы 008_Герите уже переведены в зону UTM
    assert fb == "EPSG:32638"
    assert sp_dfine.utm_for(tif, fb) == "EPSG:32638"
    # второй слой — страховка utm_for: сырая географическая CRS мимо read_utm_json → зона по центру растра
    assert sp_dfine.utm_for(tif, "EPSG:4326") == "EPSG:32638"  # 45,1° в. д. → зона 38N
    assert sp_dfine.utm_for(tif, "urn:ogc:def:crs:EPSG::4326") == "EPSG:32638"
    assert sp_dfine.utm_for(tif, "EPSG:32636") == "EPSG:32636"  # проецированная UTM.json — как раньше, без чтения растра
    assert sp_dfine.utm_for(tif, None) == "EPSG:32638"
    sc = A.Scene(tif, sp_dfine.utm_for(tif, fb), 0.5)
    w_m = 512 * 4e-6 * 111320 * np.cos(np.radians(57))
    assert abs(sc.T.a - 0.5) < 1e-9 and abs(sc.W - w_m / 0.5) / (w_m / 0.5) < 0.15 and sc.H > 100
    sc.close()


def test_gor_gate_epsg_geographic_and_projected(tmp_path):
    tif = _geo_tif(tmp_path / "g" / "04_g_SpOr_спутник" / "01_g_SpOr_спутник.tif")
    reg = SimpleNamespace(utm_json=_utm_json(tmp_path / "g" / "UTM.json", "urn:ogc:def:crs:EPSG::4326"),
                          rasters={"sp_or": [tif]})
    assert gor_gate._epsg(reg) == 32638
    reg2 = SimpleNamespace(utm_json=_utm_json(tmp_path / "g" / "UTM2.json", "urn:ogc:def:crs:EPSG::32637"),
                           rasters={"sp_or": [tif]})
    assert gor_gate._epsg(reg2) == 32637                       # проецированная — без изменений
    assert gor_gate._epsg(SimpleNamespace(utm_json=None, rasters={"sp_or": [tif]})) is None
