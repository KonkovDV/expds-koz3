"""Побитное совпадение _predict_on_device_slots с _predict_on_device (CPU-torch, тот же путь, что на CUDA)."""
import sys, os
import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from expds import infer_nn  # noqa: E402


class Fake(infer_nn.Segmenter):
    def __init__(self, tile, k, seed=0):
        self.tile, self.classes, self.device, self.tta, self.amp = tile, [f"c{i}" for i in range(k)], torch.device("cpu"), 1, False
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(k, 3, generator=g)

    def _forward(self, xb):
        # нелинейная функция с разными значениями по окнам и пикселям
        z = torch.einsum("kc,bchw->bkhw", self.w, xb)
        return torch.sigmoid(z + xb.mean(dim=(1, 2, 3), keepdim=True))


def coords_of(h, w, t, overlap=0.25):
    step = max(1, int(t * (1 - overlap)))
    ys = list(range(0, max(h - t, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - t, 0) + 1, step)) or [0]
    if ys[-1] + t < h: ys.append(h - t)
    if xs[-1] + t < w: xs.append(w - t)
    return ys, xs


@pytest.mark.parametrize("h,w,t,k,batch,overlap", [
    (128, 128, 128, 1, 8, .25), (100, 300, 128, 2, 7, .25), (517, 389, 128, 3, 16, .25),
    (1000, 777, 384, 2, 5, .25), (640, 640, 256, 1, 64, .25), (300, 1000, 128, 2, 3, .5),
    (128 + 96 * 5, 128 + 96 * 3, 128, 1, 11, .25), (129, 130, 128, 1, 4, .25),
])
def test_bit_exact(h, w, t, k, batch, overlap):
    rng = np.random.default_rng(h * 7 + w)
    img = rng.integers(0, 256, (3, h, w), dtype=np.uint8)
    m = Fake(t, k)
    pad_h, pad_w = max(0, t - h), max(0, t - w)
    x_all = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)))
    ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
    weight = np.minimum.outer(ramp, ramp); weight /= weight.max()
    ys, xs = coords_of(x_all.shape[1], x_all.shape[2], t, overlap)
    step = max(1, int(t * (1 - overlap)))
    assert infer_nn._slots_ok(ys, xs, t, step)
    ref = m._predict_on_device(x_all, [(y, x) for y in ys for x in xs], weight, batch)
    new = m._predict_on_device_slots(x_all, ys, xs, weight, batch, step)
    assert ref.dtype == new.dtype and ref.shape == new.shape
    assert np.array_equal(ref.view(np.uint32), new.view(np.uint32))
