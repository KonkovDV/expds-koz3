import numpy as np
import pytest

from centerhead import cfg_with
from centerhead.d8 import d8_array, d8_objs, d8_points, d8_target_maps, d8_vec, d8_vec_inverse
from centerhead.targets import gaussian_radius, render_targets, render_targets_strided, sigma_from_diam, size_decode


def test_shapes_and_peaks():
    objs = np.array([[10.3, 20.8, 12, 0], [40.5, 5.1, 8, 1], [63.9, 63.9, 6, 1]])
    t = render_targets(objs, 64, 64)
    assert t["hm"].shape == (2, 64, 64) and t["hm"].dtype == np.float32
    assert t["wh"].shape == (1, 64, 64) and t["off"].shape == (2, 64, 64) and t["reg_mask"].dtype == np.uint8
    assert t["hm"][0, 20, 10] == 1.0 and t["hm"][1, 5, 40] == 1.0 and t["hm"][1, 63, 63] == 1.0
    assert (t["hm"] == 1.0).sum() == 3 and t["reg_mask"].sum() == 3
    np.testing.assert_allclose(t["off"][:, 20, 10], [10.3 - 10.5, 20.8 - 20.5], atol=1e-6)
    np.testing.assert_allclose(size_decode(t["wh"][0, 20, 10]), 6.0, rtol=1e-5)
    assert t["hm"].max() <= 1.0 and t["hm"].min() >= 0.0


def test_border_tail_only():
    t = render_targets(np.array([[-1.5, 10.2, 16, 0]]), 32, 32)
    assert t["reg_mask"].sum() == 0 and t["hm"].max() < 1.0 and t["hm"][0, 10, 0] > 0.3


def test_overlap_max_and_small_wins_regression():
    objs = np.array([[10.5, 10.5, 20, 0], [12.5, 10.5, 6, 0], [10.5, 10.5, 6, 1]])
    a = render_targets(objs[:1], 32, 32)["hm"][0]
    b = render_targets(objs[1:2], 32, 32)["hm"][0]
    t = render_targets(objs, 32, 32)
    np.testing.assert_array_equal(t["hm"][0], np.maximum(a, b))
    assert t["hm"][1, 10, 10] == 1.0                                     # другой класс — свой канал
    np.testing.assert_allclose(size_decode(t["wh"][0, 10, 12]), 3.0, rtol=1e-5)


def test_sigma_rules():
    lin = cfg_with(sigma_mode="linear")
    assert sigma_from_diam(16, lin) == pytest.approx(0.15 * 16)
    assert sigma_from_diam(2, lin) == 1.0                                # sigma_min_px
    r = gaussian_radius(16, 16, 0.7)
    assert sigma_from_diam(16) == pytest.approx(max(1.0, (2 * int(r) + 1) / 6))
    assert sigma_from_diam(64) > sigma_from_diam(16)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("fl", [False, True])
def test_d8_points_match_array(k, fl):
    S = 16
    rng = np.random.default_rng(k + 4 * fl)
    for _ in range(10):
        r, c = rng.integers(S, size=2)
        a = np.zeros((S, S))
        a[r, c] = 1
        b = d8_array(a, k, fl)
        p = d8_points(np.array([[c + 0.5, r + 0.5]]), S, k, fl)[0]
        assert b[int(p[1]), int(p[0])] == 1
    v = rng.normal(size=(5, 2))
    np.testing.assert_allclose(d8_vec_inverse(d8_vec(v, k, fl), k, fl), v, atol=1e-12)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("fl", [False, True])
def test_render_after_d8_equals_d8_of_maps(k, fl):
    S = 48
    rng = np.random.default_rng(10 * k + fl)
    objs = np.column_stack([rng.uniform(3, S - 3, (6, 2)), rng.uniform(4, 12, 6), rng.integers(2, size=6)])
    t1 = render_targets(d8_objs(objs, S, k, fl), S, S)
    t2 = d8_target_maps(render_targets(objs, S, S), k, fl)
    np.testing.assert_array_equal(t1["reg_mask"], t2["reg_mask"])
    np.testing.assert_allclose(t1["hm"], t2["hm"], atol=1e-6)
    np.testing.assert_allclose(t1["off"], t2["off"], atol=1e-6)
    np.testing.assert_allclose(t1["wh"], t2["wh"], atol=1e-6)


def test_strided_targets():
    objs = np.array([[100.0, 41.0, 16, 1], [511.0, 3.0, 8, 0]])
    t = render_targets_strided(objs, 512, 4)
    assert t["hm"].shape == (2, 128, 128)
    assert t["hm"][1, 10, 25] == 1.0 and t["hm"][0, 0, 127] == 1.0
    np.testing.assert_allclose(t["off"][:, 10, 25], [0.0 - 0.5, 41 / 4 - 10.5], atol=1e-6)
    np.testing.assert_allclose(size_decode(t["wh"][0, 10, 25]), 2.0, rtol=1e-5)   # радиус 8 пикс / 4
    with pytest.raises(ValueError):
        render_targets_strided(objs, 510, 4)
