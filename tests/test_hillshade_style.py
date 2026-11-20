import numpy as np, pytest
from expds.lidar import Dem, hillshade_set, hillshade_for


def _dem():
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
    z = 2.0 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 60.0)
    z[:3, :3] = np.nan
    return Dem(dem=z, left=0.0, top=64 * 0.1, res=0.1)


def test_default_is_vendor():
    d = _dem()
    a, b = hillshade_set(d, tags=("g", "i")), hillshade_for({})(d, tags=("g", "i"))
    assert all(np.array_equal(a[k], b[k]) for k in "gi")


def test_li_match():
    d = _dem(); o = hillshade_for({"lidar": {"hillshade_style": "li_match"}})(d, tags=("g", "i"))
    v = np.isfinite(d.dem)
    assert (o["g"][~v] == 255).all() and (o["i"][~v] == 255).all()
    assert (o["g"][v].astype(int) + o["i"][v] == 253).mean() > 0.99
    assert not np.array_equal(o["g"], hillshade_set(d, tags=("g",))["g"])


def test_bad_style():
    with pytest.raises(ValueError):
        hillshade_for({"lidar": {"hillshade_style": "x"}})
