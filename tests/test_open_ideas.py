"""Выключенные приёмы: обрезка, кольцо, мёртвый пиксель, эллипс, ось стены."""
import numpy as np
from shapely.geometry import Point, box

from expds.geophys import scalar_from_bands
from expds.open_ideas import (
    choose_stable_threshold,
    component_from_peak,
    ellipse_of,
    lost_fraction,
    medial_peaks,
    repair_dead_pixels,
    ring_contrast,
    split_clipped,
    too_many_lost,
)


def test_clip_fraction_zero_keeps_every_geometry():
    geom = box(0, 0, 10, 10)
    teach, ignore = split_clipped([geom], box(0, 0, 1, 1), 0.0)
    assert teach == [geom] and ignore == []
    teach, ignore = split_clipped([geom, Point(0.2, 0.2)], box(0, 0, 1, 1), 0.15)
    assert teach == [Point(0.2, 0.2)]
    assert ignore == [geom]


def test_ring_contrast_is_absolute_and_loss_limit_is_five_percent():
    field = np.zeros((21, 21))
    field[8:13, 8:13] = 4
    bright = ring_contrast(field, 10, 10, 4)
    dark = ring_contrast(-field, 10, 10, 4)
    assert bright > 1 and abs(bright - dark) < 1e-6
    assert lost_fraction(100, 96) == 0.04
    assert not too_many_lost(100, 96)
    assert too_many_lost(100, 90)


def test_dead_pixel_and_peak_component_and_stable_threshold():
    image = np.ones((3, 3, 1))
    image[1, 1, 0] = 0
    fixed = repair_dead_pixels(image)
    assert fixed[1, 1, 0] == 1
    score = np.zeros((9, 9))
    score[3:6, 3:6] = 1
    assert component_from_peak(score, 4, 4).sum() == 9
    assert choose_stable_threshold([(0.2, 0.30), (0.4, 0.299), (0.6, 0.20)]) == 0.4


def test_ellipse_contains_center_and_wall_axis_is_not_the_hole():
    rect = box(0, 0, 10, 2)
    ell = ellipse_of(rect)
    assert ell.contains(rect.centroid)
    assert ell.area < rect.area * 1.05
    wall = np.zeros((9, 9), bool)
    wall[2:7, 2:7] = True
    wall[3:6, 3:6] = False
    peaks = medial_peaks(wall)
    assert peaks
    assert all(wall[r, c] for r, c in peaks)


def test_color_difference_is_opt_in():
    bands = np.zeros((3, 2, 2), np.float32)
    bands[0] = 5
    bands[2] = 1
    diff = scalar_from_bands(bands, "diff")
    assert np.allclose(diff, 4)
    assert np.allclose(scalar_from_bands(bands, "first"), 5)
