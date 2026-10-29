"""Метры на земле в геофизике: EPSG:3857 × cos φ (opt-in), UTM без изменений."""
import math

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from expds import crs as C
from expds import geophys


def _write(path, crs, x0, y0, step, n=200):
    a = np.full((3, n, n), 128, np.uint8)
    a[:, 90:110, 90:110] = 60                       # тёмное пятно 20×20 пикс.
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8",
                       crs=crs, transform=from_origin(x0, y0, step, step)) as w:
        w.write(a)


Y55 = 6378137.0 * math.log(math.tan(math.pi / 4 + math.radians(55.0) / 2))   # y веб-меркатора на 55° с. ш.


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


def test_load_map_3857_ground_gsd(tmp_path, monkeypatch):
    G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (EhimenNathan-derived lineage review)")
    """Карта 3857 с шагом 0,1/cos φ ед. CRS (= 0,1 м на земле) на 55°: включено — без пересэмплирования,
    gsd 0,1 м; выключено — прежнее поведение (шаг в единицах CRS)."""
    k = math.cos(math.radians(55.0))
    p = tmp_path / "m3857.tif"
    _write(p, "EPSG:3857", 3.6e6, Y55 + 10, 0.1 / k)
    monkeypatch.delenv("EXPDS_GROUND_M", raising=False)
    monkeypatch.setattr(C, "GROUND_METRES", True)
    d = G.load_map(str(p), 0.1)
    assert d["z"].shape == (200, 200)
    assert d["gsd"] == pytest.approx(0.1, rel=2e-3) and d["k"] == pytest.approx(k, rel=2e-3)
    a, T, crs, res = geophys.read_band(p)
    assert res == pytest.approx(0.1, rel=2e-3)
    monkeypatch.setattr(C, "GROUND_METRES", False)
    d0 = G.load_map(str(p), 0.1)
    # прежнее: 0,1 ед. CRS = 0,057 м на земле — карта зря увеличена в 1/cos φ ≈ 1,74 раза
    assert d0["k"] == 1.0 and d0["gsd"] == pytest.approx(0.1, rel=2e-3) and d0["z"].shape[0] == pytest.approx(349, abs=1)
    assert geophys.read_band(p)[3] == pytest.approx(0.1 / k, rel=1e-9)


def test_utm_identical(tmp_path, monkeypatch):
    G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (EhimenNathan-derived lineage review)")
    """UTM: включение ничего не меняет — массивы и шаг побайтно равны."""
    p = tmp_path / "utm.tif"
    _write(p, "EPSG:32636", 5e5, 6.1e6, 0.05)
    monkeypatch.delenv("EXPDS_GROUND_M", raising=False)
    res = {}
    for on in (False, True):
        monkeypatch.setattr(C, "GROUND_METRES", on)
        d = G.load_map(str(p), 0.1)
        res[on] = (d["z"].tobytes(), d["valid"].tobytes(), d["gsd"], tuple(d["T"]), geophys.read_band(p)[3])
    assert res[False] == res[True]
