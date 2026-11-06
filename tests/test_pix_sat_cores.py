"""PIXGBM load_map: ядра насыщения как данные (sat_core_max_m2), по умолчанию — как раньше."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (module is not in this tree)")


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


def test_sat_flag_feature_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXGBM_SAT_CORE_M2", raising=False)
    monkeypatch.delenv("PIXGBM_SAT_FLAG", raising=False)
    p = _map(tmp_path)
    rr, cc = np.array([150, 150]), np.array([200, 250])
    names, _ = G.features_for(G.load_map(p, 0.1, sat_core_max_m2=60), "maggr", rr, cc)
    assert "sat" not in names                                  # по умолчанию набор признаков прежний
    d = G.load_map(p, 0.1, sat_core_max_m2=60, sat_flag=True)
    names, X = G.features_for(d, "maggr", rr, cc)
    i = names.index("sat")
    assert X[0, i] == 1 and X[1, i] == 0 and X[0, names.index("sat1.5")] > 0
