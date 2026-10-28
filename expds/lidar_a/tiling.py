"""Tiled fully-convolutional inference with flip / rotation TTA and ramp blending (spec §3)."""
from __future__ import annotations

import numpy as np
import torch

PAD_TO = 32
OOM_TILES = (1024, 512, 256)


def tile_starts(n, tile, overlap):
    return list(range(0, max(1, n - overlap), tile - overlap))


def axis_weights(start, end, n, tile, overlap):
    """1-D blend weights (float64) of a tile covering [start, end) of an axis of length n."""
    length = end - start
    if start > 0 or end < n:
        ramp = np.minimum(np.arange(tile) + 1, overlap) / overlap
        wt = np.minimum(ramp[:length], ramp[:length][::-1])
    else:
        wt = np.ones(length)
    if start == 0:
        wt[:overlap] = 1
    if end == n:
        wt[-overlap:] = 1
    return wt


def tile_weights(ys, xs, shape, tile, overlap):
    (y0, y1), (x0, x1) = ys, xs
    return np.outer(axis_weights(y0, y1, shape[0], tile, overlap),
                    axis_weights(x0, x1, shape[1], tile, overlap)).astype(np.float32)


def view_list(tta, square):
    """[(quarter_turns, flip_dims)]: d8 on square tiles, 3 flips when TTA is on, identity otherwise."""
    if tta == "d8" and square:
        return [(k, (-1,) if mirrored else ()) for k in range(4) for mirrored in (False, True)]
    if tta:
        return [(0, ()), (0, (-1,)), (0, (-2,))]
    return [(0, ())]


def _pad_tile(part):
    h, w = part.shape[1:]
    ph, pw = (-h) % PAD_TO, (-w) % PAD_TO
    mode = "reflect" if min(h, w) > max(ph, pw) else "constant"
    return np.pad(part, ((0, 0), (0, ph), (0, pw)), mode=mode)


def _forward(model, batch):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.sigmoid(model(batch).float())


def tta_mean(model, batch, tta):
    """Mean of sigmoid outputs over the TTA views, each mapped back to the input frame. batch: (1, C, h, w)."""
    vs = view_list(tta, batch.shape[-1] == batch.shape[-2])
    total = 0
    for turns, dims in vs:
        inp = torch.rot90(batch, turns, dims=(-2, -1)) if turns else batch
        if dims:
            inp = inp.flip(dims)
        res = _forward(model, inp)
        if dims:
            res = res.flip(dims)
        total = total + (torch.rot90(res, -turns, dims=(-2, -1)) if turns else res)
    return total / len(vs)


def tile_grid(H, W, tile, overlap):
    for y0 in tile_starts(H, tile, overlap):
        for x0 in tile_starts(W, tile, overlap):
            yield (y0, min(H, y0 + tile)), (x0, min(W, x0 + tile))


@torch.no_grad()
def infer_map(model, stack, n_out, tile=1024, overlap=192, tta=True, dev="cuda"):
    """(C, H, W) float32 stack → (n_out, H, W) float32 probabilities; host (numpy) blending."""
    _, H, W = stack.shape
    acc = np.zeros((n_out, H, W), np.float32)
    norm = np.zeros((H, W), np.float32)
    for (y0, y1), (x0, x1) in tile_grid(H, W, tile, overlap):
        batch = torch.from_numpy(_pad_tile(stack[:, y0:y1, x0:x1])).float()[None].to(dev)
        pred = tta_mean(model, batch, tta)[0, :, :y1 - y0, :x1 - x0].cpu().numpy()
        wt = tile_weights((y0, y1), (x0, x1), (H, W), tile, overlap)
        acc[:, y0:y1, x0:x1] += pred * wt
        norm[y0:y1, x0:x1] += wt
    return acc / np.maximum(norm, 1e-6)


_WEIGHT_CACHE = {}


def _device_weights(ys, xs, shape, tile, overlap, dev):
    """tile_weights as a device tensor, cached by the tile's weight pattern (built on the host: identical values)."""
    (y0, y1), (x0, x1) = ys, xs
    key = (y1 - y0, x1 - x0, y0 == 0, y1 == shape[0], y0 > 0 or y1 < shape[0],
           x0 == 0, x1 == shape[1], x0 > 0 or x1 < shape[1], tile, overlap, str(dev))
    hit = _WEIGHT_CACHE.get(key)
    if hit is None:
        if len(_WEIGHT_CACHE) > 64:
            _WEIGHT_CACHE.clear()
        hit = _WEIGHT_CACHE[key] = torch.from_numpy(tile_weights(ys, xs, shape, tile, overlap)).to(dev)
    return hit


@torch.no_grad()
def infer_map_on_device(model, stack, n_out, tile=1024, overlap=192, tta=True, dev="cuda", acc_dev=None):
    """Item (5): infer_map with the blend buffers on acc_dev (default: the model's device) → torch (n_out, H, W).

    Same float32 operations in the same order as infer_map (pred·w, then +=; norm += w; acc / max(norm, 1e-6)), so the
    result is bit-equal; no per-tile device→host copy or host-side blending."""
    acc_dev = torch.device(acc_dev or dev)
    _, H, W = stack.shape
    acc = torch.zeros((n_out, H, W), dtype=torch.float32, device=acc_dev)
    norm = torch.zeros((H, W), dtype=torch.float32, device=acc_dev)
    for ys, xs in tile_grid(H, W, tile, overlap):
        (y0, y1), (x0, x1) = ys, xs
        batch = torch.from_numpy(_pad_tile(stack[:, y0:y1, x0:x1])).float()[None].to(dev)
        pred = tta_mean(model, batch, tta)[0, :, :y1 - y0, :x1 - x0].to(acc_dev)
        wt = _device_weights(ys, xs, (H, W), tile, overlap, acc_dev)
        acc[:, y0:y1, x0:x1] += pred * wt
        norm[y0:y1, x0:x1] += wt
    return acc / torch.clamp_min(norm, 1e-6)


def with_oom_ladder(run, tiles=OOM_TILES, log=print):
    """run(tile) with smaller tiles after CUDA out-of-memory."""
    for i, t in enumerate(tiles):
        try:
            return run(t)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            last = i + 1 == len(tiles)
            log(f"CUDA OOM at tile {t}" + (", giving up" if last else f", retrying at {tiles[i + 1]}"))
            if last:
                raise


def overlap_for(tile):
    return min(192, tile // 4)
