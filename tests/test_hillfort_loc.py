"""Точка городища и замена по вероятности кадра. Сдача секции не включает."""
import numpy as np
from shapely.geometry import Point

from rasterio.transform import Affine

from expds.hillfort_loc import (
    apply_scene_gate, average_maps, grid_shape, locate, phase, pooled_tokens,
    scene_probability, smooth_peak,
)


def _head(dim=4, rank=2, bias=0.0):
    return {
        "mean": np.zeros(dim, np.float32),
        "basis": np.eye(rank, dim, dtype=np.float32),
        "loc_mean": np.zeros(rank, np.float32),
        "loc_scale": np.ones(rank, np.float32),
        "coef": np.zeros(rank, np.float32),
        "intercept": bias,
    }


def test_grid_is_a_multiple_of_the_patch():
    h, w = grid_shape(1000, 400, 512)
    assert h % 16 == 0 and w % 16 == 0
    assert max(h, w) >= 512 - 16


def test_peak_is_the_bump_not_the_corner():
    score = np.zeros((8, 8), np.float32)
    score[3, 5] = 1.0
    row, col, height = smooth_peak(score, sigma=0.0)
    assert (row, col) == (3, 5)
    assert height == 1.0


def test_two_heads_average():
    tokens = np.zeros((4, 4), np.float32)
    tokens[:, 0] = 1.0
    low = _head(bias=-8.0)
    high = _head(bias=8.0)
    high["coef"] = np.array([0.0, 0.0], np.float32)
    averaged = average_maps(tokens, [low, high], 2, 2)
    assert averaged.shape == (2, 2)
    assert 0.4 < float(averaged.mean()) < 0.6


def test_scene_gate_drops_a_weak_frame_and_keeps_a_strong_point():
    old = {
        "geom": Point(0, 0),
        "class_name": "gorodishcha",
        "confidence": 0.9,
        "region_name": "a",
        "markup_type": "sp_or",
    }
    kept = {"geom": Point(1, 1), "class_name": "kurgany_povrezhdennye", "region_name": "a", "markup_type": "sp_or"}
    out = apply_scene_gate(
        [old, kept],
        [{"region": "a", "p": 0.05, "x": 4_200_000.0, "y": 7_500_000.0},
         {"region": "b", "p": 0.8, "x": 4_200_000.0, "y": 7_500_000.0}],
    )
    names = [(f["region_name"], f["class_name"]) for f in out]
    assert ("a", "gorodishcha") not in names
    assert ("a", "kurgany_povrezhdennye") in names
    added = [f for f in out if f["region_name"] == "b"]
    assert len(added) == 1
    assert added[0]["geom"].centroid.distance(Point(4_200_000.0, 7_500_000.0)) < 1.0


class _View:
    def __init__(self, tensor):
        self.last_hidden_state = tensor


class _Fake:
    config = type("C", (), {"num_register_tokens": 0})()

    def __call__(self, pixel_values):
        import torch
        batch, _, _, width = pixel_values.shape
        cells = width // 16
        tokens = torch.zeros(batch, 1 + cells, 2)
        red = pixel_values[:, 0].mean(dim=1)
        for col in range(cells):
            tokens[:, 1 + col, 0] = red[:, col * 16:(col + 1) * 16].mean(dim=1)
        return _View(tokens)


def test_flip_is_averaged_back_into_the_original_order():
    image = np.zeros((16, 32, 3), np.uint8)
    image[:, :16, 0] = 255
    tokens, gh, gw = pooled_tokens(_Fake(), image, np.zeros(3), np.ones(3))
    assert (gh, gw) == (1, 2)
    assert tokens[0, 0] > tokens[1, 0]


def test_phase_stays_empty_until_the_section_is_on():
    assert phase([], {}) == []
    assert phase([object()], {"hillfort_loc": {"enabled": False}}) == []


def test_the_circle_sits_on_the_peak_cell():
    tokens = np.zeros((4, 4), np.float32)
    tokens[3, 0] = 1.0
    head = _head()
    head["coef"] = np.array([4.0, 0.0], np.float32)
    # Четыре токена, пик в строке 1 столбце 1. Центр клетки — пиксель (24, 24).
    # Аффин уже EPSG:3857: 2 единицы на пиксель, ось y вниз.
    feat = locate(tokens, [head], 2, 2, Affine(2, 0, 0, 0, -2, 0), "a", crs="EPSG:3857")
    assert feat["threshold_group"] == "sp_dino_gor"
    assert feat["geom"].centroid.distance(Point(48.0, -48.0)) < 0.05


def test_missing_scene_head_is_not_a_number():
    tokens = np.zeros((3, 4), np.float32)
    assert np.isnan(scene_probability(tokens, None))
