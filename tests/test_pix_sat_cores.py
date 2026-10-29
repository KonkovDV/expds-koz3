"""PIXGBM load_map: ядра насыщения как данные (sat_core_max_m2), по умолчанию — как раньше."""
import numpy as np
import rasterio
from rasterio.transform import from_origin

from expds import geo_pixgbm as G


def _map(tmp_path):
    """Серая карта 0,1 м: фон-поле 255 вне съёмки (левая треть), шум 100–150, чёрное ядро 1×1 м в центре."""
    rng = np.random.default_rng(0)
    a = rng.integers(100, 150, (300, 300)).astype(np.uint8)
    a[:, :100] = 255                       # 300 м² белого поля вне съёмки, касается края кадра
    a[145:155, 195:205] = 0                # ядро сильной аномалии, 1 м²
    path = tmp_path / "01_Тест_MagGr_серый.tif"
    with rasterio.open(path, "w", driver="GTiff", width=300, height=300, count=3, dtype="uint8",
                       crs="EPSG:32636", transform=from_origin(500_000, 6_100_030, 0.1, 0.1)) as ds:
        for b in (1, 2, 3):
            ds.write(a, b)
    return path


def test_default_keeps_saturated_core_as_nodata(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGBM_SAT_CORE_M2", raising=False)
    d = G.load_map(_map(tmp_path), 0.1)
    assert not d["valid"][150, 200]
    assert not d["valid"][150, 50]


def test_option_restores_small_core_not_background(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGBM_SAT_CORE_M2", raising=False)
    d = G.load_map(_map(tmp_path), 0.1, sat_core_max_m2=60)
    assert d["valid"][150, 200] and d["z"][150, 200] < -4   # ядро — данные с крайним значением поля (≈ −5 IQR)
    assert not d["valid"][150, 50]                           # большое поле у края кадра — по-прежнему нет данных
    monkeypatch.setenv("PIXGBM_SAT_CORE_M2", "60")           # тот же переключатель для scripts/pixgbm.py
    assert G.load_map(_map(tmp_path), 0.1)["valid"][150, 200]
