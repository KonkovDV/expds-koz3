"""[raster_fix]: правки чтения снимков ae за флагом. Без секции поведение прежнее."""
import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin
from shapely.geometry import Point

from expds.raster import configure_raster_fix, open_metric, utm_json_crs, zero_field_mask
from expds.tiles import band_stats


@pytest.fixture(autouse=True)
def _reset():
    configure_raster_fix(None)
    yield
    configure_raster_fix(None)


def _site(tmp_path, crs, dtype, data, transform, nodata=None):
    d = tmp_path / "015_Тест" / "03_Тест_Ae_немецкая"
    d.mkdir(parents=True)
    (tmp_path / "015_Тест" / "UTM.json").write_text(
        json.dumps({"utm": "36N", "crs": "urn:ogc:def:crs:EPSG::32636"}), encoding="utf-8")
    p = d / "01_Тест_Ae_немецкая.tif"
    with rasterio.open(p, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata) as ds:
        ds.write(data)
    return p


def test_utm_json_crs(tmp_path):
    p = _site(tmp_path, "EPSG:32636", "uint8", np.ones((1, 8, 8), np.uint8), from_origin(500000, 6000000, 1, 1))
    assert utm_json_crs(p) == "EPSG:32636"


def test_reproject_3857_only_with_flag(tmp_path):
    # 32° в. д., 54,8° с. ш. — зона 36N; шаг файла 1,26 ед. 3857 ≈ 0,73 м на земле
    a = (np.arange(64 * 64) % 251).reshape(1, 64, 64).astype(np.uint8)
    p = _site(tmp_path, "EPSG:3857", "uint8", a, from_origin(3562223.0, 7331000.0, 1.26, 1.26))
    with open_metric(p) as ds:
        assert ds.crs.to_epsg() == 3857
    configure_raster_fix({"reproject_3857": True})
    with open_metric(p) as ds:
        assert ds.crs.to_epsg() == 32636
        assert 0.6 < abs(ds.res[0]) < 0.85
        assert ds.read(1).any()
    configure_raster_fix({"reproject_3857": True, "markups": ["sp_or"]})  # не та разметка — как было
    with open_metric(p) as ds:
        assert ds.crs.to_epsg() == 3857


def test_int16_identity(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(130, 220, size=(1, 64, 64)).astype(np.int16)
    a[:, :8] = -32768
    p = _site(tmp_path, "EPSG:32636", "int16", a, from_origin(500000, 6000000, 1, 1), nodata=-32768)
    (lo, k), = band_stats(str(p), "ae")
    assert k > 2  # прежняя растяжка p1–p99 узкого диапазона
    configure_raster_fix({"int16_identity": True})
    assert band_stats(str(p), "ae") == ((0.0, 1.0),)
    a2 = a.copy(); a2[0, 20, 20] = 400  # вне 0–255 — растяжка остаётся
    p2 = p.with_name("02_Тест_Ae_немецкая.tif")
    with rasterio.open(p, "r") as src:
        prof = src.profile
    with rasterio.open(p2, "w", **prof) as ds:
        ds.write(a2)
    assert band_stats(str(p2), "ae") != ((0.0, 1.0),)


def test_zero_field_mask_and_filter():
    import rasterio as rio

    from expds.pipeline import _outside_zero_field

    a = np.full((1, 40, 40), 120, np.uint8)
    a[:, :, :20] = 0        # чёрное поле слева
    a[:, 30, 30] = 0        # одиночный тёмный пиксель на снимке — не поле
    m = zero_field_mask(a, None, 5)
    assert m[10, 5] and m[10, 17] and not m[30, 30] and not m[10, 25]
    tr = rio.Affine(1, 0, 0, 0, -1, 40)  # пиксель (col, row) → (col, 40 − row), CRS = 3857
    found = [{"geom": Point(5.5, 40 - 10.5).buffer(1.0)},   # в поле нулей
             {"geom": Point(30.5, 40 - 30.5).buffer(1.0)}]  # на снимке
    kept = _outside_zero_field(found, m, tr, "EPSG:3857")
    assert len(kept) == 1 and kept[0] is found[1]
