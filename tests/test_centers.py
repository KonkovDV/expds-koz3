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


def test_focal_bias_starts_the_map_near_empty():
    smp = pytest.importorskip("segmentation_models_pytorch")
    from expds.centers import init_focal_bias

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2)
    init_focal_bias(model, prior=0.01)
    assert abs(float(model.segmentation_head[0].bias[0].detach()) - (-4.595)) < 0.01
    from expds.centers import focal_center_loss

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 3, 3] = 1
    missed = torch.full((1, 1, 8, 8), -6.0)
    hit = missed.clone()
    hit[0, 0, 3, 3] = 6.0
    assert float(focal_center_loss(missed, target)) > float(focal_center_loss(hit, target))


def test_unbiased_gaussian_stays_between_pixels_and_the_loss_still_sees_it():
    from expds.centers import focal_center_loss

    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 5, 5] = 1
    mask[0, 0, 5, 6] = 1
    mask[0, 0, 6, 5] = 1
    heat = gaussians_from_binary(mask, 1.5, spike=False)[0, 0]
    assert float(heat.max()) < 0.999
    assert float(heat.max()) > 0.5
    row, col = np.unravel_index(int(torch.argmax(heat)), heat.shape)
    assert abs(row - 5.333) <= 0.51
    assert abs(col - 5.333) <= 0.51
    missed = torch.full((1, 1, 16, 16), -6.0)
    hit = missed.clone()
    hit[0, 0, row, col] = 6.0
    assert float(focal_center_loss(missed, heat[None, None])) > float(focal_center_loss(hit, heat[None, None]))
    spiked = gaussians_from_binary(mask, 1.5)[0, 0]
    assert float(spiked.max()) == pytest.approx(1.0)


def test_wide_mound_gets_a_wider_gaussian():
    small = torch.zeros(1, 1, 32, 32)
    large = torch.zeros(1, 1, 32, 32)
    small[0, 0, 8:10, 8:10] = 1
    large[0, 0, 4:20, 4:20] = 1
    a = gaussians_from_binary(small, 1.0, by_radius=True)[0, 0]
    b = gaussians_from_binary(large, 1.0, by_radius=True)[0, 0]
    assert float((b > 0.5).sum()) > float((a > 0.5).sum())
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


def test_log_parabola_pulls_the_circle_toward_the_gaussian_center():
    from rasterio.transform import from_origin

    from expds.centers import subpixel_offsets

    yy, xx = np.mgrid[:24, :24]
    heat = np.exp(-((yy - 7.35) ** 2 + (xx - 11.2) ** 2) / (2 * 1.2 ** 2)).astype(np.float32)
    row, col = np.unravel_index(int(np.argmax(heat)), heat.shape)
    dy, dx = subpixel_offsets(heat, np.array([row]), np.array([col]))
    assert abs(row + float(dy[0]) - 7.35) < 0.02
    assert abs(col + float(dx[0]) - 11.2) < 0.02
    whole = centers_to_features(
        heat[None], ["kurgany_povrezhdennye"], from_origin(0, 24, 1, 1), None, 1.0,
        "region", "sp_or", "a.tif", radius_m=4.0, min_dist_m=8.0, low=0.2)
    shifted = centers_to_features(
        heat[None], ["kurgany_povrezhdennye"], from_origin(0, 24, 1, 1), None, 1.0,
        "region", "sp_or", "a.tif", radius_m=4.0, min_dist_m=8.0, low=0.2, subpixel=True)
    assert whole[0]["geom"].centroid.distance(shifted[0]["geom"].centroid) > 0.05
    assert shifted[0]["confidence"] == pytest.approx(whole[0]["confidence"])


def test_subpixel_shift_drops_the_weaker_peak_that_lands_inside_min_dist():
    from rasterio.transform import from_origin

    heat = np.zeros(32, np.float32)
    yy = np.arange(32, dtype=np.float64)
    heat += 0.9 * np.exp(-((yy - 2.45) ** 2) / (2 * 0.55 ** 2))
    heat += 0.7 * np.exp(-((yy - 10.55) ** 2) / (2 * 0.55 ** 2))
    both = centers_to_features(
        heat.reshape(1, 1, 32), ["kurgany_povrezhdennye"], from_origin(0, 0, 1, 1), None, 1.0,
        "region", "sp_or", "a.tif", radius_m=1.0, min_dist_m=8.4, low=0.2, subpixel=True)
    assert len(both) == 1
    assert both[0]["confidence"] == pytest.approx(float(heat.max()), abs=1e-5)


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
