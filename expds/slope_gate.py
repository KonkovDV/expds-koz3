"""Отсев курганов на крутом склоне. По умолчанию выключен.

Идея из полевой проверки детекторов насыпей: ложные пики часто сидят на обрыве,
а курган — на пологом месте. Режется только запасной CPU-детектор и только при
[candidates] slope_gate = true. Сданная сеть li не фильтруется.
"""
from __future__ import annotations

import numpy as np


def gentle_mask(dem, xs, ys, max_deg: float, work_res: float) -> np.ndarray:
    """True там, где уклон в точке не выше max_deg. Точка вне сетки остаётся."""
    from .candidates import to_work_grid
    from .lidar import slope_deg

    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    keep = np.ones(xs.shape[0], dtype=bool)
    if xs.size == 0:
        return keep
    z, step = to_work_grid(dem.dem, dem.res, work_res)
    valid = np.isfinite(z)
    fill = float(np.nanmean(z)) if valid.any() else 0.0
    filled = np.where(valid, z, fill).astype(np.float32)
    sl = slope_deg(filled, step)
    cols = np.rint((xs - dem.left) / step - 0.5).astype(np.int64)
    rows = np.rint((dem.top - ys) / step - 0.5).astype(np.int64)
    inside = (rows >= 0) & (cols >= 0) & (rows < sl.shape[0]) & (cols < sl.shape[1])
    sampled = np.zeros(xs.shape[0], np.float64)
    sampled[inside] = sl[rows[inside], cols[inside]]
    finite = inside & np.isfinite(sampled)
    keep[finite] = sampled[finite] <= float(max_deg)
    return keep
