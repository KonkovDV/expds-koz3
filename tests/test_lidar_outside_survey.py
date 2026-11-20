"""Вне съёмки лидара — не фон. Каналы там равны нулю до нормировки сети."""
import torch

from expds.centers import gaussians_from_binary, ignore_survey_void


def test_void_pixels_are_not_background_and_a_peak_stays():
    x = torch.zeros(1, 3, 8, 8)
    x[:, :, 2:6, 2:6] = 0.5
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 0, 0] = 1.0
    out = ignore_survey_void(x, target)
    assert float(out[0, 0, 0, 0]) == 1.0
    assert float(out[0, 0, 0, 1]) == -1.0
    assert float(out[0, 0, 4, 4]) == 0.0


def test_gaussian_keeps_the_void_mark():
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 0, :] = -1.0
    mask[0, 0, 8, 8] = 1.0
    void = mask < 0
    heat = gaussians_from_binary(mask.clamp(min=0), sigma_px=1.5, spike=True)
    heat = torch.where(void & (heat < 0.5), torch.full_like(heat, -1.0), heat)
    assert float(heat[0, 0, 0, 4]) == -1.0
    assert float(heat[0, 0, 8, 8]) == 1.0
