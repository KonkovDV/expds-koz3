"""Шероховатость и кривизна считаются из ЦМР и не входят в сданные каналы."""
import numpy as np

from expds.lidar import RELIEF_TAGS, Dem, relief_channels


def _mound() -> np.ndarray:
    yy, xx = np.mgrid[-40:41, -40:41]
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * 8.0 ** 2)).astype(np.float32)


def test_hillshade_lights_the_northwest_face():
    from expds import accel
    from expds.lidar import _hillshade_li_match

    n, res = 41, 1.0
    yy, xx = np.mgrid[:n, :n]
    x = xx.astype(np.float32)
    y = (n - 1 - yy).astype(np.float32)
    faces_nw = x - y
    faces_se = -x + y
    for fn in (lambda z: accel.hillshade(z, res, res), lambda z: _hillshade_li_match(z, res)):
        lit = int(fn(faces_nw)[n // 2, n // 2])
        dark = int(fn(faces_se)[n // 2, n // 2])
        assert lit > 200 and dark < 30


def test_hillshade_hole_is_not_a_pit_down_to_the_lowest_ground():
    from expds.lidar import hillshade_set

    z = np.full((21, 21), 10.0, np.float32)
    z[4:17, 4:17] = 14.0
    holed = z.copy()
    holed[10, 10] = np.nan
    plain = Dem(dem=z, left=0.0, top=2.1, res=0.1, crs="EPSG:32636")
    cut = Dem(dem=holed, left=0.0, top=2.1, res=0.1, crs="EPSG:32636")
    g0 = hillshade_set(plain, tags=("g",))["g"]
    g1 = hillshade_set(cut, tags=("g",))["g"]
    assert int(g1[10, 10]) == 255
    assert abs(int(g1[10, 9]) - int(g0[10, 9])) <= 2


def test_shipped_lidar_tags_stay_the_three_relief_channels():
    assert RELIEF_TAGS == ("lrm_s", "lrm_l", "slope")
    d = Dem(dem=_mound(), left=0.0, top=81.0, res=1.0, crs="EPSG:32636")
    img, _ = relief_channels(d, 1.0)
    assert img.shape[0] == 3


def test_roughness_and_curvature_follow_the_mound_and_rotate_with_it():
    mound = _mound()
    d = Dem(dem=mound, left=0.0, top=81.0, res=1.0, crs="EPSG:32636")
    img, _ = relief_channels(d, 1.0, ("rough", "curv"))
    assert img.shape == (2, 81, 81)
    assert int(img[1, 40, 40]) < 120
    assert int(img[0, 40, 28]) > int(img[0, 40, 2]) + 5
    turned = Dem(dem=np.rot90(mound), left=0.0, top=81.0, res=1.0, crs="EPSG:32636")
    img2, _ = relief_channels(turned, 1.0, ("rough", "curv"))
    assert np.array_equal(np.rot90(img[0, 2:-2, 2:-2]), img2[0, 2:-2, 2:-2])
    assert np.array_equal(np.rot90(img[1, 2:-2, 2:-2]), img2[1, 2:-2, 2:-2])
    holed = mound.copy()
    holed[0, :] = np.nan
    edge = Dem(dem=holed, left=0.0, top=81.0, res=1.0, crs="EPSG:32636")
    masked, _ = relief_channels(edge, 1.0, ("rough", "curv"))
    assert masked[:, 0, :].sum() == 0


def _noisy(noise: float, slope: float = 0.0, n: int = 160, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xx = np.arange(n)[None, :] * 0.5
    return (slope * xx + rng.normal(0, noise, (n, n))).astype(np.float32)


def test_roughness_ignores_slope_and_ranks_texture():
    def rough(z):
        img, _ = relief_channels(Dem(dem=z, left=0.0, top=80.0, res=0.5, crs="EPSG:32636"), 0.5, ("rough",))
        return float(np.median(img[0, 10:-10, 10:-10]))

    flat, steep = rough(_noisy(0.02)), rough(_noisy(0.02, slope=0.3))
    assert abs(flat - steep) <= 2                                   # уклон 17° не меняет шероховатость
    assert rough(_noisy(0.01)) + 15 < flat < rough(_noisy(0.05)) - 15


def test_curvature_marks_a_low_mound_on_a_noisy_dem_without_saturating():
    z = _noisy(0.03, slope=0.035)
    yy, xx = (np.mgrid[0:160, 0:160] + 0.5) * 0.5
    z = z + (0.3 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 1.5 ** 2))).astype(np.float32)
    img, _ = relief_channels(Dem(dem=z, left=0.0, top=80.0, res=0.5, crs="EPSG:32636"), 0.5, ("curv",))
    bg = img[0, 10:-10, 10:-10].astype(int)
    assert np.mean((bg <= 1) | (bg >= 255)) < 0.01
    assert int(img[0, 80, 80]) < int(np.median(bg)) - 40            # купол 0,3 м радиусом 3 м


def test_default_void_fill_stays_the_shipped_mean():
    z = np.arange(21, dtype=np.float32)[:, None] * np.ones((21, 21), np.float32) * 0.05
    z = z.copy()
    z[2:5, 8:13] = np.nan
    d = Dem(dem=z, left=0.0, top=21.0, res=1.0, crs="EPSG:32636")
    base, _ = relief_channels(d, 1.0, ("slope",))
    named, _ = relief_channels(d, 1.0, ("slope",), fill="mean")
    assert np.array_equal(base, named)


def test_nearest_void_fill_does_not_cut_a_cliff_into_a_gentle_slope():
    z = np.arange(21, dtype=np.float32)[:, None] * np.ones((21, 21), np.float32) * 0.05
    z = z.copy()
    z[2:5, 8:13] = np.nan
    d = Dem(dem=z, left=0.0, top=21.0, res=1.0, crs="EPSG:32636")
    mean_img, _ = relief_channels(d, 1.0, ("slope",), fill="mean")
    near_img, _ = relief_channels(d, 1.0, ("slope",), fill="nearest")
    # Ряд 5 граничит с дырой. Средняя высота съёмки далеко от местного склона.
    assert int(mean_img[0, 5, 10]) > int(near_img[0, 5, 10]) + 20
    assert int(near_img[0, 5, 10]) < 80
