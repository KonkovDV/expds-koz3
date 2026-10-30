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
    # Разброс средних секторов в метрах высоты, не в долях от отметки и не в
    # метрах плана: иначе ровный высокий склон выглядит «симметричным».
    spread = float(np.std(arr))
    return float(1.0 / (1.0 + spread / 0.05))


def profile_curvature(z: np.ndarray, row: int, col: int, res: float = 1.0, span_m: float = 3.0) -> float:
    """Профильная кривизна, 1/м. Выпуклый центр (насыпь) — положительный.

    Производные берутся по метрам плана, не по пикселям: иначе та же насыпь
    на шаге 0,5 м и 1 м даёт разные числа. Окно в несколько метров, не 3 пикселя:
    лапласиан 3×3 на шаге 0,5 м — шум высоты. В канале curv купол отрицательный;
    здесь знак обратный, чтобы порог читался как «выпукло».
    """
    if res <= 0:
        return np.nan
    half = max(1, int(round(span_m / res)))
    if not (half <= row < z.shape[0] - half and half <= col < z.shape[1] - half):
        return np.nan
    zc = float(z[row, col])
    ze, zw = float(z[row, col + half]), float(z[row, col - half])
    zn, zs = float(z[row - half, col]), float(z[row + half, col])
    zne, znw = float(z[row - half, col + half]), float(z[row - half, col - half])
    zse, zsw = float(z[row + half, col + half]), float(z[row + half, col - half])
    if not np.isfinite([zc, ze, zw, zn, zs, zne, znw, zse, zsw]).all():
        return np.nan
    dx = half * float(res)
    zx = (ze - zw) / (2.0 * dx)
    zy = (zn - zs) / (2.0 * dx)
    zxx = (ze - 2.0 * zc + zw) / (dx * dx)
    zyy = (zn - 2.0 * zc + zs) / (dx * dx)
    zxy = (zne - znw - zse + zsw) / (4.0 * dx * dx)
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
        curv = profile_curvature(z, row, col, res, span_m=near_m)
        votes = 0
        known = 0
        for ok in (np.isfinite(near) and near > 0.05,
                   np.isfinite(near) and np.isfinite(far) and near > far,
                   np.isfinite(sym) and sym >= 0.65,
                   np.isfinite(curv) and curv > 0.002):
            known += 1
            votes += int(bool(ok))
        if known and votes:
            # Только вверх. Повреждённая насыпь часто несимметрична: штраф
            # за «не купол» бил бы по самому тяжёлому классу.
            out[i] = np.float32(1.0 + 0.1 * (votes / known))
    return out
