"""circles_3857 == [circle_3857(...)] побитно (WKB) для разных CRS, включая точки на стыке зон UTM."""
import os, sys, time
import numpy as np, pytest, shapely
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from expds.raster import circle_3857, circles_3857
from expds.crs import transformer


def pts(crs, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(35.5, 36.5, n); lat = rng.uniform(44.0, 56.0, n)   # стык зон 36/37
    if crs is None:
        return rng.uniform(0, 1e5, n), rng.uniform(0, 1e5, n)
    return transformer("EPSG:4326", crs).transform(lon, lat)


@pytest.mark.parametrize("crs", ["EPSG:3857", "EPSG:4326", "EPSG:32637", "EPSG:28407", None])
@pytest.mark.parametrize("r", [4.0, 6.0, 7.5])
def test_same_wkb(crs, r):
    xs, ys = pts(crs)
    t0 = time.perf_counter(); a = [circle_3857(float(x), float(y), r, crs) for x, y in zip(xs, ys)]; ta = time.perf_counter() - t0
    t0 = time.perf_counter(); b = circles_3857(xs, ys, r, crs); tb = time.perf_counter() - t0
    assert len(a) == len(b)
    assert all(shapely.to_wkb(g) == shapely.to_wkb(h) for g, h in zip(a, b))
    print(f"{crs} r={r}: по одной {ta:.3f}s, пакетом {tb:.3f}s, x{ta / tb:.1f}")


def test_each_point_keeps_its_own_radius():
    """Свой радиус у точки не меняет WKB относительно круга по одной точке, в том числе на стыке зон."""
    xs = np.array([500_000.0, 600_000.0, 4_000_000.0])
    ys = np.array([6_100_000.0, 6_200_000.0, 7_500_000.0])
    rs = [1.0, 2.5, 3.0]
    for crs in ("EPSG:32637", "EPSG:3857"):
        one = [circle_3857(float(x), float(y), r, crs) for x, y, r in zip(xs, ys, rs)]
        batch = circles_3857(xs, ys, rs, crs)
        assert all(shapely.to_wkb(a) == shapely.to_wkb(b) for a, b in zip(one, batch))


def test_empty():
    assert circles_3857(np.array([]), np.array([]), 6.0, "EPSG:3857") == []
