"""Отражения на инференсе: обратное преобразование возвращает карту на место."""
import numpy as np
import torch

from expds.infer_nn import Segmenter


def test_tta_inverts_every_transform():
    seg = Segmenter.__new__(Segmenter)
    seg.model, seg.device, seg.amp = torch.nn.Identity(), torch.device("cpu"), False
    x = torch.from_numpy(np.random.default_rng(0).normal(size=(2, 3, 16, 12)).astype(np.float32)[..., :12, :])
    for k in (1, 2, 4, 8):
        seg.tta = k
        assert torch.allclose(seg._forward(x), torch.sigmoid(x), atol=1e-6)   # эквивариантная модель не меняется
