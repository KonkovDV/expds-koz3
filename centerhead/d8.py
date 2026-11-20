"""D8-преобразования ровно как в коде обучения: a = np.rot90(a, k, axes=(-2,-1)); если fl: a = a[..., ::-1].

Непрерывные пиксельные координаты: x — столбец, y — строка, пиксель i занимает [i, i+1), центр i+0.5.
Для квадратного кропа S×S. rot90 k=1: (x, y) → (y, S−x); вектор (dx, dy) → (dy, −dx).
flip: (x, y) → (S−x, y); вектор (dx, dy) → (−dx, dy).
"""
from __future__ import annotations

import numpy as np


def d8_array(a: np.ndarray, k: int, fl: bool) -> np.ndarray:
    a = np.rot90(a, k, axes=(-2, -1))
    return np.ascontiguousarray(a[..., ::-1] if fl else a)


def d8_points(xy: np.ndarray, S: int, k: int, fl: bool) -> np.ndarray:
    """xy (N,2) float, непрерывные пиксельные координаты в кропе S×S → те же точки после d8_array."""
    p = np.asarray(xy, np.float64).reshape(-1, 2).copy()
    for _ in range(k % 4):
        p = np.stack([p[:, 1], S - p[:, 0]], 1)
    if fl:
        p[:, 0] = S - p[:, 0]
    return p


def d8_objs(objs: np.ndarray, S: int, k: int, fl: bool) -> np.ndarray:
    """objs (N,4) [x, y, diam_px, cls] → после D8 (диаметр и класс не меняются). Рендерить цели ПОСЛЕ этого."""
    o = np.asarray(objs, np.float64).reshape(-1, 4).copy()
    if len(o):
        o[:, :2] = d8_points(o[:, :2], S, k, fl)
    return o


def torch_untta_offsets(off, k: int, fl: bool):
    """torch: off (B,2,H,W) выход сети на входе rot90(k)+flip → в исходной системе (после обратного
    flip(3) и rot90(-k) самой карты, как в predict_region кода обучения, нужно ещё повернуть компоненты)."""
    import torch
    if fl:
        off = off.flip(3)
    off = torch.rot90(off, -k, (2, 3))
    dx, dy = off[:, 0], off[:, 1]
    if fl:
        dx = -dx
    for _ in range(k % 4):
        dx, dy = -dy, dx
    return torch.stack([dx, dy], 1)
