"""Инференс растров окнами: тот же результат, что целиком, без дублей на стыках."""
import numpy as np
import rasterio
from rasterio.transform import from_origin

from conftest import UTM, X0, Y0
from expds import tiles
from expds.pipeline import _raster_with_model

SPOTS = [(20, 30), (60, 58), (100, 140), (127, 200), (190, 63), (250, 250), (57, 121)]   # строка, столбец


class BrightSeg:
    """Модель-заглушка: вероятность класса = яркость первого канала."""
    classes = ["kurgany_tselye"]
    res = 0.5
    meta = {"in_channels": 3, "max_upsample": 2.0}

    def predict(self, img):
        return img[:1].astype(np.float32) / 255.0


def write_raster(path, dtype=np.uint8):
    a = np.zeros((3, 300, 270), dtype)
    for r, c in SPOTS:
        a[:, r:r + 6, c:c + 6] = 250
    with rasterio.open(path, "w", driver="GTiff", height=300, width=270, count=3, dtype=a.dtype,
                       transform=from_origin(X0, Y0 + 150, 0.5, 0.5), crs=UTM) as ds:
        ds.write(a)
    return path


def test_working_res_caps_upsampling():
    assert tiles.working_res(0.2, 0.5) == 0.5
    assert tiles.working_res(1.6, 0.5) == 0.8
    assert tiles.working_res(1.6, 0.5, max_upsample=4.0) == 0.5


def test_band_stats_stretch_whole_raster(tmp_path):
    p = write_raster(tmp_path / "a16.tif", np.int16)
    st = tiles.band_stats(str(p), "ae")
    assert st is not None and len(st) == 3
    a = rasterio.open(p).read()
    u = tiles.to_uint8(a[:, :50, :50], "ae", st)
    v = tiles.to_uint8(a, "ae", st)[:, :50, :50]
    assert np.array_equal(u, v)                         # окно растягивается так же, как весь снимок
    assert tiles.band_stats(str(write_raster(tmp_path / "a8.tif")), "ae") is None


def test_windowed_inference_matches_whole(tmp_path):
    p = write_raster(tmp_path / "ae.tif")
    whole = _raster_with_model(BrightSeg(), p, "r", "ae", block=4096, overlap=256)
    small = _raster_with_model(BrightSeg(), p, "r", "ae", block=64, overlap=16)
    assert len(whole) == len(SPOTS) == len(small)
    cw = sorted((round(f["geom"].centroid.x, 1), round(f["geom"].centroid.y, 1)) for f in whole)
    cs = sorted((round(f["geom"].centroid.x, 1), round(f["geom"].centroid.y, 1)) for f in small)
    assert cw == cs
