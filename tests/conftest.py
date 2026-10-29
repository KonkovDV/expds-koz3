"""Общие фикстуры: синтетические LAS, растры и папки регионов (без конкурсных данных)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference" / "lib"))

# Файлы сдачи, которых нет в git (веса обучены на конкурсных данных; expds/geo_pixgbm.py — на проверке
# происхождения, docs/MERGE_2026-10-29.md). Тесты сборки архива с ними пропускаются, если файлов нет.
NOT_IN_GIT = ("models/rerank_arch_lgbm.txt", "models/pixgbm/all/poly.txt", "models/pixgbm/all/finds.txt",
              "expds/geo_pixgbm.py")
_MISSING = [p for p in NOT_IN_GIT if not (ROOT / p).is_file()]
requires_submission_files = pytest.mark.skipif(
    bool(_MISSING), reason=f"submission files not kept in git: {', '.join(_MISSING)}")

# Синтетический участок в UTM 36N (около 33° в. д., 55° с. ш.)
UTM = "EPSG:32636"
X0, Y0 = 500_000.0, 6_100_000.0
MOUNDS = [(30.0, 40.0, 1.2, 5.0), (70.0, 60.0, 0.8, 4.0), (45.0, 85.0, 1.0, 6.0)]  # x, y, высота, радиус (м)


def mound_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Плавный рельеф + курганы-гауссианы (координаты от левого нижнего угла, м)."""
    z = 150.0 + 0.02 * x + 0.01 * y
    for mx, my, h, r in MOUNDS:
        z = z + h * np.exp(-((x - mx) ** 2 + (y - my) ** 2) / (2 * (r / 2) ** 2))
    return z


def write_las(path: Path, n: int = 120_000, seed: int = 0, ground_class: int = 2, wkt: bool = True) -> Path:
    """LAS 1.4 с точками земли (класс ground_class) и «растительностью» (класс 5)."""
    import laspy
    from pyproj import CRS

    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 100, n)
    # дыра в покрытии (пустоты DEM) и несколько точек растительности
    keep = ~((x > 80) & (x < 85) & (y > 10) & (y < 14))
    x, y = x[keep], y[keep]
    z = mound_surface(x, y)
    cls = np.full(x.size, ground_class, np.uint8)
    veg = rng.random(x.size) < 0.1
    z = np.where(veg, z + rng.uniform(2, 10, x.size), z)
    cls[veg] = 5
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([X0, Y0, 0.0])
    if wkt:
        hdr.add_crs(CRS.from_user_input(UTM))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = x + X0, y + Y0, z
    las.classification = cls
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))
    return path


def write_mag_tif(path: Path, res: float = 0.25, size_m: float = 40.0, seed: int = 1) -> tuple[Path, list]:
    """Карта магниторазведки с диполями (точечные аномалии) и одним «пятном»."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    n = int(size_m / res)
    yy, xx = np.mgrid[0:n, 0:n] * res
    a = rng.normal(0, 1.0, (n, n)).astype(np.float32)
    points = [(8.0, 9.0), (25.0, 30.0), (31.0, 12.0)]
    for px, py in points:
        a += 25 * np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 0.3 ** 2))
        a -= 12 * np.exp(-((xx - px - 0.8) ** 2 + (yy - py) ** 2) / (2 * 0.3 ** 2))
    a += 6 * ((xx - 15) ** 2 / 4 + (yy - 20) ** 2 / 2.25 < 1)   # площадная аномалия ~4,7 м²
    tr = from_origin(X0, Y0 + size_m, res, res)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       transform=tr, crs=UTM) as ds:
        ds.write(a, 1)
    # координаты точек в CRS (y растёт вверх; строки вниз)
    pts = [(X0 + px + res / 2, Y0 + size_m - py - res / 2) for px, py in points]
    return path, pts


@pytest.fixture
def val_dir(tmp_path: Path) -> Path:
    """Папка «валидационной выборки» из двух регионов."""
    r1 = tmp_path / "input" / "region_0001"
    write_las(r1 / "01_region_0001_La_точки" / "01_region_0001_La_точки.las")
    (r1 / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    r2 = tmp_path / "input" / "region_0002"
    write_mag_tif(r2 / "07_region_0002_MagGr_карты" / "01_region_0002_MagGr.tif")
    return tmp_path / "input"


@pytest.fixture
def official_metric() -> str | None:
    """Путь к официальному compute_metrics.py, если задан EXPDS_METRIC."""
    import os

    p = os.environ.get("EXPDS_METRIC")
    return p if p and Path(p).exists() else None
