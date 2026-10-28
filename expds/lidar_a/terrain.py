"""Relief stack at 0.5 m for the LiDAR networks (spec §2), computed with torch on any device."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NET_CELL = 0.5
LRM_SCALES_M = (2.0, 5.0, 10.0, 25.0)
SUN_AZIMUTHS = (315.0, 45.0)
SUN_ALTITUDE = 35.0
HORIZON_M = 10.0
N_DIRECTIONS = 8
WINDOW_PX = 3072
WINDOW_MARGIN_PX = 384
BASE_CHANNELS = 9
EXTRA_CHANNELS = 12
# the 12-channel stack inserts the three openness channels before `valid`: 9-ch = these planes of the 12-ch one
BASE_OF_EXTRA = [0, 1, 2, 3, 4, 5, 6, 7, 11]


def n_channels(extra):
    return EXTRA_CHANNELS if extra else BASE_CHANNELS


def _taps(sigma, like):
    k = max(int(3 * sigma) | 1, 3)
    offs = torch.arange(k, device=like.device, dtype=like.dtype) - k // 2
    w = torch.exp(-offs ** 2 / (2 * sigma ** 2))
    return w / w.sum(), k // 2


def smooth(img, sigma):
    """Separable Gaussian, edge-replicated borders, horizontal pass first. img: (1, 1, h, w)."""
    w, half = _taps(sigma, img)
    k = w.numel()
    img = F.conv2d(F.pad(img, (half, half, 0, 0), mode="replicate"), w.reshape(1, 1, 1, k))
    return F.conv2d(F.pad(img, (0, 0, half, half), mode="replicate"), w.reshape(1, 1, k, 1))


def smooth_wide(img, sigma):
    """Gaussian for wide sigma: average-pool by q = ⌊σ/6⌋, blur σ/q, bilinear back up (σ ≤ 12: plain smooth)."""
    if sigma <= 12:
        return smooth(img, sigma)
    q = int(sigma // 6)
    h, w = img.shape[-2:]
    small = F.avg_pool2d(F.pad(img, (0, (-w) % q, 0, (-h) % q), mode="replicate"), q)
    small = smooth(small, sigma / q)
    return F.interpolate(small, scale_factor=q, mode="bilinear", align_corners=False)[..., :h, :w]


def _resample(dtm, valid, cell, dev):
    """→ (z, v) as (1, 1, h, w) tensors at NET_CELL."""
    f = NET_CELL / cell
    h, w = int(round(dtm.shape[0] / f)), int(round(dtm.shape[1] / f))
    fi = int(round(f))
    whole = f > 1 and abs(f - fi) < 1e-6
    if whole:
        ey, ex = (-dtm.shape[0]) % fi, (-dtm.shape[1]) % fi
        if ey or ex:
            dtm = np.pad(dtm, ((0, ey), (0, ex)), mode="edge")
            valid = np.pad(valid, ((0, ey), (0, ex)))
    z = torch.from_numpy(np.ascontiguousarray(dtm)).to(dev)[None, None]
    v = torch.from_numpy(valid.astype(np.float32)).to(dev)[None, None]
    if abs(f - 1) <= 1e-6:
        return z, v
    if whole:
        return F.avg_pool2d(z, fi)[..., :h, :w], F.avg_pool2d(v, fi)[..., :h, :w]
    shrink = f > 1
    return (F.interpolate(z, size=(h, w), mode="area" if shrink else "bilinear"),
            F.interpolate(v, size=(h, w), mode="area" if shrink else "nearest"))


def _central_diff(zs, r):
    gx = F.pad((zs[..., :, 2:] - zs[..., :, :-2]) / (2 * r), (1, 1, 0, 0), mode="replicate")
    gy = F.pad((zs[..., 2:, :] - zs[..., :-2, :]) / (2 * r), (0, 0, 1, 1), mode="replicate")
    return gx, gy


def _hillshade(slope, gx, gy, azimuth_deg):
    az, alt = np.radians(azimuth_deg), np.radians(SUN_ALTITUDE)
    aspect = torch.atan2(-gx, gy)
    return np.sin(alt) * torch.cos(slope) + np.cos(alt) * torch.sin(slope) * torch.cos(az - aspect)


def _laplacian(z, r):
    stencil = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=z.device, dtype=z.dtype).reshape(1, 1, 3, 3)
    return F.conv2d(F.pad(smooth(z, 1.0 / r), (1, 1, 1, 1), mode="replicate"), stencil) / (r * r)


def _horizons(z, r):
    """Positive / negative openness and sky-view factor (horizon search, HORIZON_M, N_DIRECTIONS) → 3 planes."""
    reach = max(2, int(round(HORIZON_M / r)))
    stride = max(1, reach // 10)
    h, w = z.shape[-2:]
    big = F.pad(z, (reach,) * 4, mode="replicate")
    pos_sum = neg_sum = sky_sum = 0
    for theta in np.arange(N_DIRECTIONS) * 2 * np.pi / N_DIRECTIONS:
        s, c = np.sin(theta), np.cos(theta)
        up = torch.full_like(z, -np.pi / 2)
        down = torch.full_like(z, -np.pi / 2)
        for step in range(1, reach + 1, stride):
            oy, ox = int(round(step * s)), int(round(step * c))
            dist = np.hypot(ox, oy) * r
            if dist == 0:
                continue
            other = big[..., reach + oy:reach + oy + h, reach + ox:reach + ox + w]
            elev = torch.atan((other - z) / dist)
            up = torch.maximum(up, elev)
            down = torch.maximum(down, -elev)
        pos_sum = pos_sum + (np.pi / 2 - up)
        neg_sum = neg_sum + (np.pi / 2 - down)
        sky_sum = sky_sum + torch.sin(up.clamp(min=0))
    pos, neg = pos_sum / N_DIRECTIONS, neg_sum / N_DIRECTIONS
    svf = 1 - sky_sum / N_DIRECTIONS
    return [((pos - np.pi / 2) / 0.1).clamp(-4, 4), ((neg - np.pi / 2) / 0.1).clamp(-4, 4),
            ((svf - 0.97) / 0.03).clamp(-4, 4)]


@torch.no_grad()
def relief_stack(dtm, valid, cell, extra=False, unclip=False, dev="cuda"):
    """dtm (H, W) float32 at `cell` m → (C, h, w) float32 at NET_CELL and valid (h, w) bool; C = 9 or 12 (extra)."""
    z, v = _resample(dtm, valid, cell, dev)
    r = NET_CELL
    planes = []
    for scale in LRM_SCALES_M:
        lrm = (z - smooth_wide(z, scale / r)) / 0.5
        planes.append(torch.asinh(lrm) if unclip else lrm.clamp(-4, 4))
    gx, gy = _central_diff(smooth(z, 0.75), r)
    slope = torch.atan(torch.sqrt(gx ** 2 + gy ** 2))
    planes.append(slope / 0.5)
    for az in SUN_AZIMUTHS:
        planes.append(_hillshade(slope, gx, gy, az) * 2 - 1)
    planes.append((_laplacian(z, r) / 0.2).clamp(-4, 4))
    if extra:
        planes += _horizons(z, r)
    inside = (v > 0.5).float()
    planes.append(inside * 2 - 1)
    stack = torch.cat(planes, 1)[0] * inside[0]
    return stack.cpu().numpy().astype(np.float32), inside[0, 0].bool().cpu().numpy()


def relief_stack_windowed(dtm, valid, cell, extra=False, unclip=False, dev="cuda", window=WINDOW_PX, margin=WINDOW_MARGIN_PX):
    """relief_stack over overlapping windows (bounded device memory); equal to the one-shot stack on window cores."""
    f = NET_CELL / cell
    H, W = int(round(dtm.shape[0] / f)), int(round(dtm.shape[1] / f))
    if H * W <= window * window:
        return relief_stack(dtm, valid, cell, extra, unclip, dev)
    out = np.zeros((n_channels(extra), H, W), np.float32)
    vout = np.zeros((H, W), bool)
    for wy in range(0, H, window):
        for wx in range(0, W, window):
            ay, ax = max(0, wy - margin), max(0, wx - margin)
            by, bx = min(H, wy + window + margin), min(W, wx + window + margin)
            sy, sx = int(round(ay * f)), int(round(ax * f))
            ny, nx = int(round((by - ay) * f)), int(round((bx - ax) * f))
            dpart, vpart = dtm[sy:sy + ny, sx:sx + nx], valid[sy:sy + ny, sx:sx + nx]
            short_y, short_x = ny - dpart.shape[0], nx - dpart.shape[1]
            if short_y or short_x:
                dpart = np.pad(dpart, ((0, short_y), (0, short_x)), mode="edge")
                vpart = np.pad(vpart, ((0, short_y), (0, short_x)))
            st, vv = relief_stack(np.ascontiguousarray(dpart), vpart, cell, extra, unclip, dev)
            oy, ox = wy - ay, wx - ax
            ch = min(window, H - wy, st.shape[1] - oy)
            cw = min(window, W - wx, st.shape[2] - ox)
            out[:, wy:wy + ch, wx:wx + cw] = st[:, oy:oy + ch, ox:ox + cw]
            vout[wy:wy + ch, wx:wx + cw] = vv[oy:oy + ch, ox:ox + cw]
    return out, vout
