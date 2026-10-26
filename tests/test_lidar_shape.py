"""Шероховатость и кривизна считаются из ЦМР и не входят в сданные каналы."""
import numpy as np

from expds.lidar import RELIEF_TAGS, Dem, relief_channels


def _mound() -> np.ndarray:
    yy, xx = np.mgrid[-40:41, -40:41]
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * 8.0 ** 2)).astype(np.float32)


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
