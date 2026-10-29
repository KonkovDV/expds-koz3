"""Цели CenterNet: гауссовы теплокарты по классам + плотные карты размера/смещения.

objs: float (N,4) [x, y, diam_px, cls], x/y — непрерывные пиксельные координаты кропа (x = (X−left)/step,
y = (top−Y)/step, как region_targets у bot-9), cls ∈ {0: tselye, 1: povr}.
Центр-пиксель: cx = floor(x), cy = floor(y) (= round(x−0,5) у bot-9 вне точных половинок).
Смещение: off = (x − (cx+0,5), y − (cy+0,5)) ∈ [−0,5, 0,5).
"""
from __future__ import annotations

import math

import numpy as np

from .config import DEFAULTS


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """CenterNet (xingyizhou/CenterNet, src/lib/utils/image.py gaussian_radius) — формула оставлена как в оригинале."""
    a1, b1, c1 = 1.0, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 ** 2 - 4 * a1 * c1))) / 2
    a2, b2, c2 = 4.0, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(max(0.0, b2 ** 2 - 4 * a2 * c2))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(0.0, b3 ** 2 - 4 * a3 * c3))) / 2
    return min(r1, r2, r3)


def sigma_from_diam(diam_px: float, cfg: dict = DEFAULTS) -> float:
    if cfg["sigma_mode"] == "linear":                 # как render_heat bot-9
        s = cfg["sigma_k"] * diam_px
    elif cfg["sigma_mode"] == "centernet":            # CenterNet: radius=max(0,int(r)), sigma=(2r+1)/6
        r = max(0, int(gaussian_radius(diam_px, diam_px, cfg["min_overlap"])))
        s = (2 * r + 1) / 6.0
    else:
        raise ValueError(cfg["sigma_mode"])
    return max(cfg["sigma_min_px"], float(s))


def size_encode(radius_px, cfg: dict = DEFAULTS):
    r = np.maximum(np.asarray(radius_px, np.float64), 1e-3)
    return np.log(r) if cfg["size_mode"] == "log_r" else r


def size_decode(v, cfg: dict = DEFAULTS):
    v = np.asarray(v, np.float64)
    return np.exp(np.clip(v, -5, 8)) if cfg["size_mode"] == "log_r" else np.maximum(v, 0)


def render_targets(objs: np.ndarray, H: int, W: int, n_cls: int = 2, cfg: dict = DEFAULTS) -> dict:
    """→ {"hm": f32 (n_cls,H,W) ∈ [0,1], пик ровно 1 в центр-пикселе; "wh": f32 (1,H,W) — код радиуса (size_mode);
    "off": f32 (2,H,W);
    "reg_mask": u8 (H,W) — 1 в центр-пикселях объектов, центр которых внутри кропа}.
    Перекрытия: np.maximum по теплокарте; размер/смещение в общем пикселе — у меньшего объекта (рендер от больших к меньшим).
    Объекты с центром вне кропа дают только хвост гауссианы (без положительного пикселя и без регрессии)."""
    hm = np.zeros((n_cls, H, W), np.float32)
    wh = np.zeros((1, H, W), np.float32)
    off = np.zeros((2, H, W), np.float32)
    rm = np.zeros((H, W), np.uint8)
    o = np.asarray(objs, np.float64).reshape(-1, 4)
    for x, y, d, c in o[np.argsort(-o[:, 2], kind="stable")]:
        c = int(c)
        if not (0 <= c < n_cls) or not (np.isfinite(x) and np.isfinite(y)):
            continue
        s = sigma_from_diam(float(d), cfg)
        cx, cy = int(math.floor(x)), int(math.floor(y))
        R = int(math.ceil(3 * s))
        x0, x1, y0, y1 = max(0, cx - R), min(W, cx + R + 1), max(0, cy - R), min(H, cy + R + 1)
        if x0 < x1 and y0 < y1:
            gx = np.arange(x0, x1) - cx
            gy = np.arange(y0, y1) - cy
            g = np.exp(-(gy[:, None] ** 2 + gx[None, :] ** 2) / (2 * s * s)).astype(np.float32)
            np.maximum(hm[c, y0:y1, x0:x1], g, out=hm[c, y0:y1, x0:x1])
        if 0 <= cx < W and 0 <= cy < H:
            hm[c, cy, cx] = 1.0
            wh[0, cy, cx] = size_encode(d / 2.0, cfg)
            off[0, cy, cx] = x - (cx + 0.5)
            off[1, cy, cx] = y - (cy + 0.5)
            rm[cy, cx] = 1
    return {"hm": hm, "wh": wh, "off": off, "reg_mask": rm}


def render_targets_strided(objs: np.ndarray, S: int, stride: int = DEFAULTS["stride"], n_cls: int = 2,
                           cfg: dict = DEFAULTS) -> dict:
    """objs в пикселях входного кропа S×S (0,5 м) → цели на сетке выхода (S/stride)², шаг res·stride.
    x, y, diam делятся на stride; sigma_min_px — в пикселях выхода; wh — радиус в пикселях выхода (код size_mode);
    off — в пикселях выхода. Вызывать ПОСЛЕ d8_objs (D8 на входном кропе)."""
    if S % stride:
        raise ValueError(f"crop {S} не делится на stride {stride}")
    o = np.asarray(objs, np.float64).reshape(-1, 4).copy()
    o[:, :3] /= stride
    return render_targets(o, S // stride, S // stride, n_cls, cfg)
