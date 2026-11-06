"""pick_rasters: по умолчанию серая копия; opt-in «hp_energy» — копия с большей мелкой структурой."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (module is not in this tree)")


def _write(path, a):
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0], count=3, dtype="uint8",
                       crs="EPSG:32636", transform=from_origin(500_000, 6_100_030, 0.1, 0.1)) as ds:
        for b in (1, 2, 3):
            ds.write(a, b)


def test_pick_rule(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGBM_PICK", raising=False)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:200, :200]
    smooth = (120 + 20 * np.sin(xx / 60.0) + rng.normal(0, 0.3, (200, 200))).astype(np.uint8)   # почти без структуры
    busy = smooth.copy()
    for _ in range(80):                                                                          # мелкие аномалии
        y, x = rng.integers(5, 195, 2); busy[y - 2:y + 3, x - 2:x + 3] = rng.integers(40, 220)
    g, c = tmp_path / "01_Т_MagGr_серый.tif", tmp_path / "02_Т_MagGr_цвет.tif"
    _write(g, smooth); _write(c, busy)
    assert G.pick_rasters([g, c]) == [g]
    assert G.pick_rasters([g, c], "hp_energy", "maggr") == [c]
    monkeypatch.setenv("PIXGBM_PICK", "hp_energy")
    assert G.pick_rasters([g, c], None, "maggr") == [c]
    assert G.pick_rasters([g], None, "maggr") == [g]
