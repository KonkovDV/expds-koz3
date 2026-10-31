"""Отражения и мультимасштаб на инференсе (bot-8).

Флаги OFF (tta=1, ms_scales=[1.0]) должны давать тот же путь, что до bot-8.
"""
from __future__ import annotations

import numpy as np
import torch

from expds.infer_nn import Ensemble, Segmenter


def _bare_seg(tile: int = 16) -> Segmenter:
    seg = Segmenter.__new__(Segmenter)
    seg.model = torch.nn.Identity()
    seg.device = torch.device("cpu")
    seg.amp = False
    seg.tta = 1
    seg.ms_scales = [1.0]
    seg.tile = tile
    seg.classes = ["kurgany_tselye"]
    seg.meta = {"in_channels": 3, "classes": seg.classes, "tile": tile}
    return seg


def test_tta_inverts_every_transform():
    seg = _bare_seg()
    x = torch.from_numpy(
        np.random.default_rng(0).normal(size=(2, 3, 16, 12)).astype(np.float32)[..., :12, :]
    )
    for k in (1, 2, 4, 8):
        seg.tta = k
        assert torch.allclose(seg._forward(x), torch.sigmoid(x), atol=1e-6)


def test_ms_scales_default_inactive():
    assert Segmenter._ms_active(None) is False
    assert Segmenter._ms_active([]) is False
    assert Segmenter._ms_active([1.0]) is False
    assert Segmenter._ms_active([1.0, 1.0]) is True  # два прохода — активен
    assert Segmenter._ms_active([0.75, 1.0, 1.25]) is True


def test_resize_chw_roundtrip_shape():
    arr = np.random.default_rng(1).integers(0, 255, size=(3, 40, 60), dtype=np.uint8)
    up = Segmenter._resize_chw(arr, 80, 120)
    assert up.shape == (3, 80, 120)
    back = Segmenter._resize_chw(up, 40, 60)
    assert back.shape == (3, 40, 60)
    assert back.dtype == np.uint8


def test_predict_ms_off_matches_single_scale(monkeypatch):
    """ms_scales=[1.0] вызывает ровно _predict_at_scale (один проход)."""
    seg = _bare_seg(tile=8)
    calls = []

    def fake(img, overlap=0.25, batch=None):
        calls.append(img.shape)
        c, h, w = img.shape
        return np.full((len(seg.classes), h, w), 0.5, np.float32)

    monkeypatch.setattr(seg, "_predict_at_scale", fake)
    img = np.zeros((3, 16, 16), np.uint8)
    seg.ms_scales = [1.0]
    out = seg.predict(img)
    assert len(calls) == 1 and calls[0] == (3, 16, 16)
    assert out.shape == (1, 16, 16)


def test_predict_ms_averages_scales(monkeypatch):
    seg = _bare_seg(tile=8)
    calls = []

    def fake(img, overlap=0.25, batch=None):
        calls.append(img.shape)
        c, h, w = img.shape
        return np.full((len(seg.classes), h, w), 0.4, np.float32)

    monkeypatch.setattr(seg, "_predict_at_scale", fake)
    img = np.zeros((3, 20, 20), np.uint8)
    seg.ms_scales = [0.5, 1.0]
    out = seg.predict(img)
    assert len(calls) == 2
    assert out.shape == (1, 20, 20)
    assert np.allclose(out, 0.4)


def test_ensemble_ms_scales_propagates():
    a, b = _bare_seg(), _bare_seg()
    a.meta = b.meta = {
        "classes": a.classes, "res": 0.5, "tile": 16, "in_channels": 3,
        "li_tags": None, "li_source": None, "dem_res": None, "dem_fill": None, "markup": "sp_or",
    }
    # bypass __init__ meta checks by building Ensemble carefully
    ens = Ensemble.__new__(Ensemble)
    ens.members = [a, b]
    ens.meta = a.meta
    ens.classes = a.classes
    ens.res = 0.5
    ens.tile = 16
    ens.keep = None
    ens.only = None
    ens.ms_scales = [0.75, 1.0, 1.25]
    assert a.ms_scales == [0.75, 1.0, 1.25]
    assert b.ms_scales == [0.75, 1.0, 1.25]
