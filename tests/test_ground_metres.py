"""Метры на земле в геофизике: EPSG:3857 × cos φ (opt-in), UTM без изменений."""
import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from expds import crs as C


def _write(path, crs, x0, y0, step, n=200):
    a = np.full((3, n, n), 128, np.uint8)
    a[:, 90:110, 90:110] = 60                       # тёмное пятно 20×20 пикс.
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8",
                       crs=crs, transform=from_origin(x0, y0, step, step)) as w:
        w.write(a)


Y55 = 6378137.0 * math.log(math.tan(math.pi / 4 + math.radians(55.0) / 2))   # y веб-меркатора на 55° с. ш.


def test_ground_metres_off_by_default(monkeypatch):
    monkeypatch.setattr(C, "GROUND_METRES", None)
    monkeypatch.delenv("EXPDS_GROUND_M", raising=False)
    assert C.ground_metres_on() is False
    assert C.ground_scale("EPSG:3857", bounds=(0, 0, 1, 1)) == 1.0


def test_ground_scale(monkeypatch):
    monkeypatch.setattr(C, "GROUND_METRES", None)
    monkeypatch.delenv("EXPDS_GROUND_M", raising=False)
    b = (3.6e6, Y55 - 10, 3.6e6 + 10, Y55 + 10)
    assert C.ground_scale("EPSG:3857", b) == 1.0                       # выключено по умолчанию
    assert C.ground_scale("EPSG:3857", b, enable=True) == pytest.approx(math.cos(math.radians(55.0)), 1e-6)
    assert C.ground_scale("EPSG:32636", (5e5, 6.1e6, 5.1e5, 6.11e6), enable=True) == 1.0
    monkeypatch.setenv("EXPDS_GROUND_M", "1")
    assert C.ground_scale("EPSG:3857", b) == pytest.approx(0.573576, 1e-5)
    monkeypatch.setattr(C, "GROUND_METRES", False)                     # конфиг важнее env
    assert C.ground_scale("EPSG:3857", b) == 1.0


