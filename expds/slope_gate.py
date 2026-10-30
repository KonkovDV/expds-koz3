"""Отсев курганов на крутом склоне. По умолчанию выключен.

Идея из полевой проверки детекторов насыпей: ложные пики часто сидят на обрыве,
а курган — на пологом месте. Режется только запасной CPU-детектор и только при
[candidates] slope_gate = true. Сданная сеть li не фильтруется.
"""
from __future__ import annotations

import numpy as np


def gentle_mask(dem, xs, ys, max_deg: float, work_res: float) -> np.ndarray:
    """True там, где уклон не выше max_deg. Точка вне сетки и точка без измерения остаются."""
    from .candidates import to_work_grid
    from .lidar import slope_deg

    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    keep = np.ones(xs.shape[0], dtype=bool)
    if xs.size == 0:
        return keep
    z, step = to_work_grid(dem.dem, dem.res, work_res)
    valid = np.isfinite(z)
    # Ноль только чтобы ядро Хорна не увидело NaN. Окно, которое касается дырки,
    # после расчёта стирается: заливка средней высотой рисует ложный обрыв.
    filled = np.where(valid, z, 0).astype(np.float32)
    sl = slope_deg(filled, step).astype(np.float64, copy=True)
    if not valid.all():
        bad = ~valid
        pad = np.pad(bad, 1, mode="constant", constant_values=False)
        touch = np.zeros(bad.shape, dtype=bool)
        h, w = bad.shape
        for dr in range(3):
            for dc in range(3):
                touch |= pad[dr:dr + h, dc:dc + w]
        sl[touch] = np.nan
    cols = np.floor((xs - dem.left) / step).astype(np.int64)
    rows = np.floor((dem.top - ys) / step).astype(np.int64)
    inside = (rows >= 0) & (cols >= 0) & (rows < sl.shape[0]) & (cols < sl.shape[1])
    sampled = np.zeros(xs.shape[0], np.float64)
    sampled[inside] = sl[rows[inside], cols[inside]]
    finite = inside & np.isfinite(sampled)
    keep[finite] = sampled[finite] <= float(max_deg)
    return keep
