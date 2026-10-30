"""Насыпь отличается от склона по TPI, симметрии и кривизне. Сдача этот множитель не включает."""
import numpy as np

from expds.dome import priors, profile_curvature, symmetry, tpi


def _mound(n=81, res=0.5):
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (yy - n // 2) ** 2 + (xx - n // 2) ** 2
    return (1.2 * np.exp(-r2 / (2 * 6.0 ** 2))).astype(np.float32), res


def _slope(n=81):
    yy, xx = np.mgrid[0:n, 0:n]
    return (0.04 * xx).astype(np.float32)


def test_mound_is_a_local_dome():
    z, res = _mound()
    c = z.shape[0] // 2
    assert tpi(z, c, c, radius_px=10) > 0.2
    assert symmetry(z, c, c, radius_px=12) > 0.7
    assert profile_curvature(z, c, c, res) > 0


def test_curvature_does_not_change_when_the_cell_size_changes():
    def gauss(res):
        sigma, h, span = 4.0, 1.2, 40.0
        n = int(round(span / res))
        n += (n + 1) % 2
        c = n // 2
        yy, xx = np.mgrid[0:n, 0:n]
        x = (xx - c) * res
        y = (yy - c) * res
        z = (h * np.exp(-(x * x + y * y) / (2 * sigma * sigma))).astype(np.float32)
        return z, c

    fine, cf = gauss(0.5)
    coarse, cc = gauss(1.0)
    a = profile_curvature(fine, cf, cf, 0.5, span_m=3.0)
    b = profile_curvature(coarse, cc, cc, 1.0, span_m=3.0)
    assert a > 0.02 and b > 0.02
    assert abs(a - b) / max(a, b) < 0.15


def test_slope_is_not_a_dome():
    z = _slope()
    c = z.shape[0] // 2
    assert abs(tpi(z, c, c, radius_px=10)) < 0.05
    assert symmetry(z, c, c, radius_px=12) < 0.7


def test_prior_keeps_one_without_evidence_and_lifts_a_mound():
    z, res = _mound()
    n = z.shape[0]
    left, top = 0.0, n * res
    center = (left + (n // 2 + 0.5) * res, top - (n // 2 + 0.5) * res)
    mound = priors(z, res, [center[0]], [center[1]], left, top)
    slope = priors(_slope(n), res, [center[0]], [center[1]], left, top)
    assert mound[0] > slope[0]
    assert slope[0] == np.float32(1.0)
    assert mound[0] <= np.float32(1.1)
    wedge = z + np.linspace(0, 0.4, n, dtype=np.float32)[None, :]
    irregular = priors(wedge, res, [center[0]], [center[1]], left, top)
    assert irregular[0] >= np.float32(1.0)
    empty = priors(z, res, [], [], left, top)
    assert empty.shape == (0,)
