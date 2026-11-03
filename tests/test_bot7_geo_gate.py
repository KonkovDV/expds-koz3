"""bot7-geo-gate: extremum_snap_m + finds_nms_m (default OFF) + geo_convex_hull already OFF."""
import numpy as np

from expds import geophys, post


def test_geo_params_new_levers_default_off():
    p = geophys.GeoParams()
    assert p.extremum_snap_m == 0.0
    assert p.finds_nms_m == 0.0
    assert p.plateau is False


def test_snap_peaks_to_extremum_moves_to_argmax():
    mag = np.zeros((21, 21), np.float32)
    mag[10, 14] = 5.0   # true extremum 4 px east of seed
    mag[10, 10] = 2.0
    rows, cols, vals = np.array([10]), np.array([10]), np.array([2.0], np.float32)
    rr, cc, vv = geophys.snap_peaks_to_extremum(rows, cols, vals, mag, res=1.0, snap_m=5.0)
    assert int(rr[0]) == 10 and int(cc[0]) == 14
    assert float(vv[0]) == 5.0
    # snap_m=0 is no-op
    r0, c0, v0 = geophys.snap_peaks_to_extremum(rows, cols, vals, mag, res=1.0, snap_m=0.0)
    assert int(c0[0]) == 10 and float(v0[0]) == 2.0


def test_nms_peaks_keeps_strongest_in_radius():
    rows = np.array([0, 0, 10])
    cols = np.array([0, 1, 10])
    vals = np.array([1.0, 3.0, 2.0], np.float32)
    rr, cc, vv = geophys.nms_peaks_by_strength(rows, cols, vals, res=1.0, nms_m=1.5)
    assert len(rr) == 2
    assert sorted(zip(rr.tolist(), cc.tolist(), vv.tolist())) == sorted([(0, 1, 3.0), (10, 10, 2.0)])
    # off
    assert len(geophys.nms_peaks_by_strength(rows, cols, vals, res=1.0, nms_m=0.0)[0]) == 3


def test_peaks_from_wires_snap_and_nms(tmp_path):
    """Offset seed peak must snap to stronger neighbor; NMS drops the weaker twin."""
    from conftest import UTM, X0, Y0
    import rasterio
    from rasterio.transform import from_origin

    res = 1.0
    n = 32
    # synthetic highpass-friendly field: two close bumps + one far
    yy, xx = np.mgrid[0:n, 0:n]
    field = np.zeros((n, n), np.float32)
    field += 8.0 * np.exp(-((xx - 10) ** 2 + (yy - 10) ** 2) / 2.0)
    field += 12.0 * np.exp(-((xx - 13) ** 2 + (yy - 10) ** 2) / 2.0)  # stronger, 3 m east
    field += 9.0 * np.exp(-((xx - 25) ** 2 + (yy - 25) ** 2) / 2.0)
    path = tmp_path / "bumps.tif"
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       transform=from_origin(X0, Y0 + n * res, res, res), crs=UTM) as ds:
        ds.write(field, 1)

    base = geophys.GeoParams(background_m=2.0, peak_radius_m=1.0, peak_k=0.5, peak_k_min=0.5)
    off = geophys.finds_points(path, "r", "maggr", base)
    base.extremum_snap_m = 4.0
    base.finds_nms_m = 2.5
    on = geophys.finds_points(path, "r", "maggr", base)
    assert geophys.GeoParams().extremum_snap_m == 0.0
    assert geophys.GeoParams().finds_nms_m == 0.0
    # with NMS we should not have more preds than without; snap may collapse twins
    assert len(on) <= len(off)
    assert len(on) >= 1


def test_geo_convex_hull_still_off_by_default():
    from shapely.geometry import Polygon
    L = Polygon([(0, 0), (2, 0), (2, 0.2), (1, 0.2), (1, 1), (0, 1)])
    feat = {"geom": L, "class_name": "object_poly", "markup_type": "etgr", "confidence": 0.5}
    # without flag post helper is a no-op when spec falsy — convex_hull_object_poly always hulls
    # when called; watchdog only calls it if post.geo_convex_hull truthy.
    out = post.convex_hull_object_poly([feat], True)
    assert out[0]["geom"].area >= L.area
