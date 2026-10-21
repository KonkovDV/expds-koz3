"""Середина плато F1, а не острый максимум на одном пороге."""
import numpy as np

from expds.metric import plateau_index


def test_plateau_picks_the_middle_of_a_flat_top():
    f1 = np.array([0.1, 0.50, 0.50, 0.50, 0.10])
    assert plateau_index(f1, tol=0.002) == 2


def test_plateau_ignores_a_spike_below_the_tolerance():
    f1 = np.array([0.20, 0.510, 0.509, 0.508, 0.30])
    assert plateau_index(f1, tol=0.002) == 2


def test_group_plateau_moves_off_the_sharp_peak():
    """Длинный хвост почти не меняет F1. Острый максимум отрезает его, середина плато — нет."""
    from collections import Counter

    from expds.metric import ClassCurve, sweep_groups

    n_hit, n_fp, n_gt = 100, 80, 1000
    conf = np.linspace(0.99, 0.01, n_hit + n_fp)
    hit = np.array([True] * n_hit + [False] * n_fp)
    curve = ClassCurve(conf=conf, hit=hit, pred_region=["a"] * conf.size, n_gt=n_gt,
                       gt_regions=Counter({"a": n_gt}), pred_group=["li"] * conf.size)
    sharp = sweep_groups({"kurgany_povrezhdennye": curve}, pick="max")
    flat = sweep_groups({"kurgany_povrezhdennye": curve}, pick="plateau")
    sharp_g = sharp["kurgany_povrezhdennye"]["groups"]["li"]
    flat_g = flat["kurgany_povrezhdennye"]["groups"]["li"]
    assert sharp_g["kept"] == n_hit
    assert flat_g["kept"] > sharp_g["kept"]
    assert flat["kurgany_povrezhdennye"]["f1"] >= sharp["kurgany_povrezhdennye"]["f1"] - 0.002

