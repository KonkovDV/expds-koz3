"""geo_snap: poly_mode="peaks" — слитая компонента с двумя максимумами даёт две точки, режим pole — одну."""
import numpy as np
import pytest
from affine import Affine

G = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py is held out of main (module is not in this tree)")


def _two_blobs(gap_m):
    gsd = 0.1
    yy, xx = np.mgrid[0:80, 0:160] * gsd
    P = np.maximum(np.exp(-((xx - 4) ** 2 + (yy - 4) ** 2) / 8), 0.95 * np.exp(-((xx - 4 - gap_m) ** 2 + (yy - 4) ** 2) / 8))
    d = {"gsd": gsd, "T": Affine(gsd, 0, 400000, 0, -gsd, 6000000), "crs": "EPSG:32636"}
    return P.astype(np.float32), d


def test_peaks_split_merged_component():
    P, d = _two_blobs(3.0)
    pole = G.poly_candidates(P, d, "r", "maggr", "f", 0.5)
    pk = G.poly_candidates(P, d, "r", "maggr", "f", 0.5, mode="peaks", peak_r_m=1.0)
    assert len(pole) == 1
    assert len(pk) == 2
    assert {c["threshold_group"] for c in pk} == {"maggr_pix"}
    assert max(c["confidence"] for c in pk) > 0.99


def test_peaks_nms_radius_merges_close_maxima():
    P, d = _two_blobs(1.5)
    assert len(G.poly_candidates(P, d, "r", "maggr", "f", 0.5, mode="peaks", peak_r_m=2.0)) == 1
