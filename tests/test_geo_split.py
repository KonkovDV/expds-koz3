"""Разрезание слипшихся object_poly водоразделом и круги в полюсе недоступности."""
import numpy as np
from rasterio.transform import from_origin

from expds import geophys
from expds.geophys import GeoParams, label_poles, polys_from, split_merged


def _two_blobs(gap_px=0, res=0.1):
    """Две гауссовы ямы, слипшиеся по порогу."""
    yy, xx = np.mgrid[0:80, 0:120]
    f = 3 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 12 ** 2))
    f += 3 * np.exp(-((xx - 75 - gap_px) ** 2 + (yy - 40) ** 2) / (2 * 12 ** 2))
    return f.astype(np.float32)


def test_split_field_separates_merged_pair():
    f = _two_blobs()
    lab = (f >= 1.0).astype(np.int32)
    from scipy.ndimage import label; assert label(lab)[1] == 1
    p = GeoParams(poly_k=1.0, split="field", split_h=0.1, min_area_m2=1.0)
    out, n = split_merged(lab, f, 0.1, p)
    assert n == 2
    assert (out[40, 40] != out[40, 75]) and out[40, 40] > 0 and out[40, 75] > 0
    assert ((out > 0) == (lab > 0)).all()


def test_split_keeps_single_blob_whole():
    yy, xx = np.mgrid[0:80, 0:80]
    f = (3 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 12 ** 2))).astype(np.float32)
    lab = (f >= 1.0).astype(np.int32)
    for mode, h in (("field", 0.1), ("dt", 0.25)):
        out, n = split_merged(lab, f, 0.1, GeoParams(poly_k=1.0, split=mode, split_h=h, min_area_m2=1.0))
        assert n == 1 and ((out > 0) == (lab > 0)).all()


def test_split_dt_cuts_dumbbell_neck():
    lab = np.zeros((60, 120), np.int32)
    yy, xx = np.mgrid[0:60, 0:120]
    lab[((xx - 30) ** 2 + (yy - 30) ** 2) < 20 ** 2] = 1
    lab[((xx - 90) ** 2 + (yy - 30) ** 2) < 20 ** 2] = 1
    lab[27:33, 30:90] = 1
    out, n = split_merged(lab, np.zeros(lab.shape, np.float32), 0.1, GeoParams(split="dt", split_h=0.5, min_area_m2=1.0))
    assert n == 2 and out[30, 30] != out[30, 90]


def test_small_pieces_absorbed():
    f = _two_blobs()
    lab = (f >= 1.0).astype(np.int32)
    out, n = split_merged(lab, f, 0.1, GeoParams(poly_k=1.0, split="field", split_h=0.1, min_area_m2=1e4))
    assert n == 1


def test_label_poles_inside_and_central():
    lab = np.zeros((50, 50), np.int32)
    lab[10:41, 5:16] = 1          # прямоугольник 31×11
    lab[20:31, 25:46] = 2
    ks, rows, cols = label_poles(lab)
    for k, r, c in zip(ks, rows, cols):
        assert lab[r, c] == k
    assert abs(rows[0] - 25) <= 1 and abs(cols[0] - 10) <= 1


def test_polys_from_pole_circles():
    f = _two_blobs()
    tr = from_origin(500000.0, 6100000.0, 0.1, 0.1)
    base = dict(background_m=8, smooth_m=0.1, poly_k=1.0, min_area_m2=1.0, max_area_m2=1000.0)
    plain = polys_from(f, tr, "EPSG:32636", 0.1, "r", "maggr", "s", GeoParams(**base))
    split = polys_from(f, tr, "EPSG:32636", 0.1, "r", "maggr", "s",
                       GeoParams(**base, split="field", split_h=0.1, pole_circle_m=0.05))
    assert len(plain) == 1 and len(split) == 2
    for ft in split:
        assert ft["geom"].geom_type == "Polygon" and plain[0]["geom"].contains(ft["geom"].centroid)


def test_default_is_unchanged():
    p = GeoParams()
    assert p.split == "none" and p.pole_circle_m == 0.0
    assert geophys.SPLITS[0] == "none"


def test_split_spec_from_config():
    p, _ = geophys.params_for({"maggr": {"object_poly": {"split_spec": "field:0.2", "pole_circle_m": 0.05}}},
                              "maggr", "object_poly")
    assert p.split == "field" and p.split_h == 0.2 and p.pole_circle_m == 0.05
    p, _ = geophys.params_for({}, "maggr", "object_poly")
    assert p.split == "none" and p.pole_circle_m == 0.0
