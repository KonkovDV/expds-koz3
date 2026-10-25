"""Центры пятен насыщения на пересвеченной карте MagGr (geophys.plateau_centers, plateau = true)."""
import numpy as np
import rasterio
from conftest import UTM, X0, Y0
from rasterio.transform import from_origin

from expds import geophys

CENTERS = [(10.0, 10.0), (25.0, 12.0), (15.0, 28.0)]      # м от левого верхнего угла


def saturated_map(path, res: float = 0.1, size_m: float = 40.0):
    """Белый фон ровно 255 (как 78 % пикселей серой карты Гнёздова, Терраса 1), тёмные аномалии,
    ядро срезано в 0 радиусом около 3,9 м — больше допуска метрики 2 м."""
    rng = np.random.default_rng(0)
    n = int(size_m / res)
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) * res
    field = np.zeros((n, n))
    for cx, cy in CENTERS:
        field += 600 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
    a = 255 - field
    shade = (a > 0) & (a < 250)
    a[shade] += rng.normal(0, 1.5, int(shade.sum()))
    a[a >= 250] = 255                                         # фон — край шкалы
    a = np.rint(np.clip(a, 0, 255)).astype(np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="uint8",
                       transform=from_origin(X0, Y0 + size_m, res, res), crs=UTM) as ds:
        ds.write(a, 1)
    return path, [(X0 + cx, Y0 + size_m - cy) for cx, cy in CENTERS]


def hits(feats, truth, tol=2.0):
    from expds.crs import WEB_MERCATOR, to_crs

    pts = [to_crs(f["geom"], WEB_MERCATOR, UTM).centroid for f in feats]
    return sum(any(np.hypot(p.x - x, p.y - y) <= tol for p in pts) for x, y in truth)


def test_plateau_centers_sit_in_the_saturated_cores(tmp_path):
    path, truth = saturated_map(tmp_path / "sat.tif")
    a, tr, _crs, res = geophys.read_band(path)
    hp = geophys.highpass(geophys.robust_z(a), res, 2.0)
    rows, cols, vals, mask = geophys.plateau_centers(a, hp, res, min_strength=1.0)
    assert len(rows) == 3 and mask.any()                      # фон крупнее 60 м² — не пятно
    from expds.raster import pixel_to_xy

    xs, ys = pixel_to_xy(tr, rows, cols)
    for x, y in truth:
        assert min(np.hypot(np.asarray(xs) - x, np.asarray(ys) - y)) < 0.5
    assert (vals > 0).all()


def test_plateau_mode_finds_the_cores_that_edge_peaks_miss(tmp_path):
    path, truth = saturated_map(tmp_path / "sat.tif")
    p = geophys.GeoParams(background_m=2.0, peak_radius_m=2.0, peak_k=1.0)
    plain = geophys.finds_points(path, "r", "maggr", p)
    p.plateau = True
    with_cores = geophys.finds_points(path, "r", "maggr", p)
    assert hits(with_cores, truth) == 3
    assert hits(with_cores, truth) > hits(plain, truth)


def test_plateau_off_by_default_and_harmless_on_float_maps(tmp_path):
    from conftest import write_mag_tif

    path, _ = write_mag_tif(tmp_path / "f.tif")
    p = geophys.GeoParams(background_m=2.0, peak_radius_m=2.0, peak_k=1.0)
    base = geophys.finds_points(path, "r", "maggr", p)
    p.plateau = True
    same = geophys.finds_points(path, "r", "maggr", p)
    assert geophys.GeoParams().plateau is False
    assert len(same) == len(base)                               # без срезанных краёв шкалы пятен нет


def test_core_below_half_percent_of_the_map_is_kept():
    """Ядро на большом листе занимает меньше 0,5 % пикселей. Порог доли его выбрасывал."""
    res = 0.5
    n = 240                                                      # 120 м, 57600 пикселей
    a = np.full((n, n), 180, np.float32)
    a[4, 4] = 0                                                  # крапинка 0,25 м²
    yy, xx = np.mgrid[0:n, 0:n]
    core = (xx - 180) ** 2 + (yy - 180) ** 2 <= (8) ** 2         # радиус 4 м, около 50 м²
    a[core] = 0
    share = float((a == 0).mean())
    assert share < 0.005
    hp = np.zeros_like(a)
    hp[core] = 0
    rim = ndimage_ring(core)
    hp[rim] = 30
    kept, *_ = geophys.plateau_centers(a, hp, res, min_strength=1.0)
    dropped, *_ = geophys.plateau_centers(a, hp, res, min_share=0.005, min_strength=1.0)
    assert len(kept) == 1 and len(dropped) == 0


def ndimage_ring(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_dilation(mask, iterations=2) & ~mask
