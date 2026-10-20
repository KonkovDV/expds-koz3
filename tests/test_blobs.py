"""Кандидаты-пятна на оптике: светлые и тёмные пятна находятся, окна не дают дублей."""
import numpy as np
import rasterio
from conftest import UTM, X0, Y0
from rasterio.transform import from_origin

from expds import blobs

SPOTS = [(60, 60, 40), (150, 220, -40), (240, 90, 35), (300, 330, -30)]   # строка, столбец, Δяркости


def write_photo(path):
    rng = np.random.default_rng(0)
    a = 120 + rng.normal(0, 4, (400, 420))
    yy, xx = np.mgrid[0:400, 0:420]
    for r, c, d in SPOTS:
        a += d * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 6.0 ** 2))
    a = np.clip(a, 1, 255).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=400, width=420, count=1, dtype="uint8",
                       transform=from_origin(X0, Y0 + 200, 0.5, 0.5), crs=UTM) as ds:
        ds.write(a, 1)
    return path


def test_bright_and_dark_spots_found_once(tmp_path):
    p = write_photo(tmp_path / "ae.tif")
    whole = blobs.detect(p, blobs.BlobParams(min_contrast=1.0))
    small = blobs.detect(p, blobs.BlobParams(min_contrast=1.0, block=128))
    for c in (whole, small):
        assert c["crs"] == UTM and c["features"].shape[1] == 7
        for r, col, d in SPOTS:
            x, y = X0 + (col + 0.5) * 0.5, Y0 + 200 - (r + 0.5) * 0.5
            dist = np.hypot(c["x"] - x, c["y"] - y)
            k = int(np.argmin(dist))
            assert dist[k] < 2.0 and np.sign(c["features"][k, 1]) == np.sign(d)   # знак отклика = светлое/тёмное
    assert len(small["x"]) == len(whole["x"])
