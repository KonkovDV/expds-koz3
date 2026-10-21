"""Карта центров: два пятна остаются двумя пиками, пик становится кругом."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from expds.centers import centers_to_features, gaussians_from_binary, heatmap_counts


def test_two_mounds_do_not_merge_into_the_gap():
    mask = torch.zeros(1, 1, 40, 40)
    mask[0, 0, 5:9, 5:9] = 1
    mask[0, 0, 28:32, 30:34] = 1
    heat = gaussians_from_binary(mask, 2.0)[0, 0].numpy()
    assert heat[6, 6] > 0.9
    assert heat[29, 31] > 0.9
    assert heat[18, 18] < 0.05


def test_empty_mask_stays_empty():
    heat = gaussians_from_binary(torch.zeros(2, 3, 8, 8), 1.5)
    assert float(heat.sum()) == 0.0


def test_matching_peaks_score_one_and_a_shift_does_not():
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 8, 8] = 1
    tp, fp, fn = heatmap_counts(target, target, 2)
    assert (int(tp), int(fp), int(fn)) == (1, 0, 0)
    shifted = torch.zeros_like(target)
    shifted[0, 0, 8, 24] = 1
    tp, fp, fn = heatmap_counts(shifted, target, 2)
    assert (int(tp), int(fp), int(fn)) == (0, 1, 1)


def test_peak_becomes_a_circle_around_itself():
    from rasterio.transform import from_origin

    prob = np.zeros((1, 30, 30), np.float32)
    prob[0, 10, 20] = 0.8
    feats = centers_to_features(
        prob, ["kurgany_povrezhdennye"], from_origin(0, 30, 1, 1), None, 1.0,
        "region", "sp_or", "a.tif", radius_m=6.0, min_dist_m=8.0, low=0.2)
    assert len(feats) == 1
    geom = feats[0]["geom"]
    center = geom.centroid
    assert abs(center.x - 20.5) < 0.2
    assert abs(center.y - 19.5) < 0.2
    # Метрика берёт representative_point, не центроид. У круга она внутри и рядом с пиком.
    assert geom.representative_point().distance(center) < 2.0
    assert abs(feats[0]["geom"].area - math.pi * 36) / (math.pi * 36) < 0.15
    assert feats[0]["confidence"] == pytest.approx(0.8)


def test_center_loss_penalises_a_missed_peak():
    from expds.centers import center_loss

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 3] = 1
    missed = torch.full((1, 1, 8, 8), -8.0)
    hit = missed.clone()
    hit[0, 0, 3, 3] = 8.0
    assert float(center_loss(missed, target)) > float(center_loss(hit, target))
