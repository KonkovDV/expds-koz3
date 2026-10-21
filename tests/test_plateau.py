"""Середина плато F1, а не острый максимум на одном пороге."""
import numpy as np

from expds.metric import plateau_index


def test_plateau_picks_the_middle_of_a_flat_top():
    f1 = np.array([0.1, 0.50, 0.50, 0.50, 0.10])
    assert plateau_index(f1, tol=0.002) == 2


def test_plateau_ignores_a_spike_below_the_tolerance():
    f1 = np.array([0.20, 0.510, 0.509, 0.508, 0.30])
    assert plateau_index(f1, tol=0.002) == 2
