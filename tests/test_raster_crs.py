"""CRS растра: безымянная LOCAL_CS и растр без CRS получают UTM.json участка (crs.usable_raster_crs)."""
from pathlib import Path

import numpy as np
import rasterio
from conftest import UTM, write_mag_tif
from rasterio.crs import CRS

from expds import geophys
from expds.crs import to_3857_ok, usable_raster_crs

LOCAL = 'LOCAL_CS["unnamed",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'


def copy_with_crs(src: Path, dst: Path, crs) -> Path:
    with rasterio.open(src) as ds:
        a, tr = ds.read(1), ds.transform
    with rasterio.open(dst, "w", driver="GTiff", height=a.shape[0], width=a.shape[1], count=1, dtype="float32",
                       transform=tr, crs=crs) as out:
        out.write(a, 1)
    return dst


def test_raster_crs_rules():
    assert to_3857_ok(UTM) and not to_3857_ok(None)
    utm_box = (500_000.0, 6_100_000.0, 500_040.0, 6_100_040.0)
    assert usable_raster_crs(UTM, utm_box, "EPSG:32637") == UTM                    # объявленная рабочая — как есть
    with rasterio.open(_tmp_local()) as ds:
        declared = ds.crs.to_string()
    assert not to_3857_ok(declared)
    assert usable_raster_crs(declared, utm_box, UTM) == UTM                         # LOCAL_CS + метры UTM → UTM.json
    assert usable_raster_crs(None, utm_box, UTM) == UTM                             # без CRS — так же
    assert usable_raster_crs(declared, utm_box, None) == declared                   # без UTM.json угадывать нечего
    assert usable_raster_crs(None, (0.0, 0.0, 160.0, 160.0), UTM) is None            # пиксели — не угадываются
    assert usable_raster_crs(None, (3_670_000.0, 7_370_000.0, 3_671_000.0, 7_371_000.0), None) == "EPSG:3857"


def _tmp_local() -> Path:
    import tempfile

    d = Path(tempfile.mkdtemp())
    src, _ = write_mag_tif(d / "utm.tif")
    return copy_with_crs(src, d / "local.tif", CRS.from_wkt(LOCAL))


def test_local_cs_map_gives_the_same_finds_as_the_utm_original(tmp_path):
    src, _ = write_mag_tif(tmp_path / "utm.tif")
    loc = copy_with_crs(src, tmp_path / "local.tif", CRS.from_wkt(LOCAL))
    ref = geophys.finds_points(src, "r", "maggr", geophys.GeoParams())
    got = geophys.finds_points(loc, "r", "maggr", geophys.GeoParams(), fallback_crs=UTM)
    assert len(got) == len(ref) > 0
    a = np.array([[f["geom"].centroid.x, f["geom"].centroid.y] for f in ref])
    b = np.array([[f["geom"].centroid.x, f["geom"].centroid.y] for f in got])
    assert np.allclose(a, b, atol=1e-6)
    assert geophys.map_footprint(loc, UTM) is not None and geophys.map_footprint(loc) is None
