"""Признаки насыпи по рельефу: два масштаба TPI, симметрия, кривизна.

Свой расчёт по окну DEM. В сдаче выключен: множитель вешается только на запасной
CPU-детектор курганов и только при [candidates] dome = true. Восемь признаков
обученной модели не меняются.
"""
from __future__ import annotations

import numpy as np

DOME_NAMES = ("tpi_near", "tpi_far", "symmetry", "profile_curv")


def _disk_mean(z: np.ndarray, row: int, col: int, radius_px: int) -> float:
    r0, r1 = max(0, row - radius_px), min(z.shape[0], row + radius_px + 1)
    c0, c1 = max(0, col - radius_px), min(z.shape[1], col + radius_px + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    dist = (yy - row) ** 2 + (xx - col) ** 2
    patch = z[r0:r1, c0:c1]
    mask = (dist > 0) & (dist <= radius_px ** 2) & np.isfinite(patch)
    if not np.any(mask):
        return np.nan
    return float(patch[mask].mean())


def tpi(z: np.ndarray, row: int, col: int, radius_px: int) -> float:
    """Высота центра минус средняя высота окрестности. Насыпь даёт плюс."""
    if radius_px < 1 or not (0 <= row < z.shape[0] and 0 <= col < z.shape[1]):
        return np.nan
    center = z[row, col]
    if not np.isfinite(center):
        return np.nan
    around = _disk_mean(z, row, col, radius_px)
    if not np.isfinite(around):
        return np.nan
    return float(center - around)


def symmetry(z: np.ndarray, row: int, col: int, radius_px: int, sectors: int = 8) -> float:
    """Близко к 1, когда сектора кольца одной высоты. Круглая насыпь выше косогора."""
    if radius_px < 2:
        return np.nan
    inner = max(1, radius_px // 3)
    r0, r1 = max(0, row - radius_px), min(z.shape[0], row + radius_px + 1)
    c0, c1 = max(0, col - radius_px), min(z.shape[1], col + radius_px + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    dy, dx = yy - row, xx - col
    dist2 = dy ** 2 + dx ** 2
    patch = z[r0:r1, c0:c1]
    ring = (dist2 >= inner ** 2) & (dist2 <= radius_px ** 2) & np.isfinite(patch)
    if int(ring.sum()) < sectors:
        return np.nan
    ang = np.arctan2(dy, dx)
    bins = np.floor((ang + np.pi) / (2 * np.pi) * sectors).astype(np.int64)
    bins = np.clip(bins, 0, sectors - 1)
    means = []
    flat_p, flat_b, flat_m = patch.ravel(), bins.ravel(), ring.ravel()
    for s in range(sectors):
        take = flat_m & (flat_b == s)
        if np.any(take):
            means.append(float(flat_p[take].mean()))
    if len(means) < sectors // 2:
        return np.nan
    arr = np.asarray(means, np.float64)
    # Разброс в метрах, не относительно абсолютной высоты: иначе ровный склон
    # выглядит «симметричным» только потому, что стоит высоко.
    spread = float(np.std(arr))
    return float(1.0 / (1.0 + spread / 0.05))


def profile_curvature(z: np.ndarray, row: int, col: int) -> float:
    """Профильная кривизна по окну 3×3. Выпуклый центр (насыпь) — положительный."""
    if not (1 <= row < z.shape[0] - 1 and 1 <= col < z.shape[1] - 1):
        return np.nan
    win = z[row - 1:row + 2, col - 1:col + 2].astype(np.float64)
    if not np.isfinite(win).all():
        return np.nan
    zx = (win[1, 2] - win[1, 0]) / 2.0
    zy = (win[0, 1] - win[2, 1]) / 2.0
    zxx = win[1, 2] - 2.0 * win[1, 1] + win[1, 0]
    zyy = win[0, 1] - 2.0 * win[1, 1] + win[2, 1]
    zxy = (win[0, 2] - win[0, 0] - win[2, 2] + win[2, 0]) / 4.0
    p2q2 = zx * zx + zy * zy
    if p2q2 < 1e-8:
        return float(-(zxx + zyy) / 2.0)
    return float(-(zx * zx * zxx + 2.0 * zx * zy * zxy + zy * zy * zyy) / (p2q2 * (1.0 + p2q2) ** 1.5))


def priors(z: np.ndarray, res: float, xs: np.ndarray, ys: np.ndarray, left: float, top: float,
           near_m: float = 5.0, far_m: float = 40.0) -> np.ndarray:
    """Множитель уверенности по каждому центру. Ниже 1 не опускается."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    out = np.ones(xs.shape[0], np.float32)
    if xs.size == 0 or res <= 0:
        return out
    near_px = max(1, int(round(near_m / res)))
    far_px = max(near_px + 1, int(round(far_m / res)))
    for i, (x, y) in enumerate(zip(xs, ys)):
        col = int(round((x - left) / res - 0.5))
        row = int(round((top - y) / res - 0.5))
        near = tpi(z, row, col, near_px)
        far = tpi(z, row, col, far_px)
        sym = symmetry(z, row, col, near_px)
        curv = profile_curvature(z, row, col)
        votes = 0
        known = 0
        for ok in (np.isfinite(near) and near > 0.05,
                   np.isfinite(near) and np.isfinite(far) and near > far,
                   np.isfinite(sym) and sym >= 0.65,
                   np.isfinite(curv) and curv > 1e-3):
            known += 1
            votes += int(bool(ok))
        if known and votes:
            # Только вверх. Повреждённая насыпь часто несимметрична: штраф
            # за «не купол» бил бы по самому тяжёлому классу.
            out[i] = np.float32(1.0 + 0.1 * (votes / known))
    return out
