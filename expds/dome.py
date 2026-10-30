"""Признаки насыпи по рельефу: два масштаба TPI, симметрия, кривизна.

Свой расчёт по окну DEM. В сдаче выключен: множитель вешается только на запасной
CPU-детектор курганов и только при [candidates] dome = true. Восемь признаков
обученной модели не меняются.
"""
from __future__ import annotations

import numpy as np

DOME_NAMES = ("tpi_near", "tpi_far", "symmetry", "k_min")


def _disk_mean(z: np.ndarray, row: int, col: int, radius_px: float) -> float:
    rad = float(radius_px)
    if not np.isfinite(rad) or rad < 1.0:
        return np.nan
    r = int(np.ceil(rad - 1e-9))
    r0, r1 = max(0, row - r), min(z.shape[0], row + r + 1)
    c0, c1 = max(0, col - r), min(z.shape[1], col + r + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    dist = (yy - row) ** 2 + (xx - col) ** 2
    patch = z[r0:r1, c0:c1]
    # Граница круга — в метрах, переведённых в пиксели, а не в округлённом радиусе.
    mask = (dist > 0) & (dist <= rad * rad) & np.isfinite(patch)
    if int(mask.sum()) < 0.5 * np.pi * rad * rad:
        return np.nan
    return float(patch[mask].mean())


def tpi(z: np.ndarray, row: int, col: int, radius_px: float) -> float:
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


def symmetry(z: np.ndarray, row: int, col: int, radius_px: float, sectors: int = 8) -> float:
    """Близко к 1, когда сектора кольца одной высоты. Круглая насыпь выше косогора."""
    rad = float(radius_px)
    if not np.isfinite(rad) or rad < 2.0:
        return np.nan
    inner = max(1.0, rad / 3.0)
    r = int(np.ceil(rad - 1e-9))
    r0, r1 = max(0, row - r), min(z.shape[0], row + r + 1)
    c0, c1 = max(0, col - r), min(z.shape[1], col + r + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    dy, dx = yy - row, xx - col
    dist2 = dy ** 2 + dx ** 2
    patch = z[r0:r1, c0:c1]
    ring = (dist2 >= inner ** 2) & (dist2 <= rad * rad) & np.isfinite(patch)
    annulus = np.pi * (rad * rad - inner ** 2)
    if int(ring.sum()) < max(sectors, 0.5 * annulus):
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
    # Кольцо без сектора — не симметрия: обрывок склона у края может выглядеть ровным.
    if len(means) < sectors:
        return np.nan
    arr = np.asarray(means, np.float64)
    # Разброс средних секторов в метрах высоты, не в долях от отметки и не в
    # метрах плана: иначе ровный высокий склон выглядит «симметричным».
    spread = float(np.std(arr))
    return float(1.0 / (1.0 + spread / 0.05))


def _principal_min(zx: float, zy: float, zxx: float, zyy: float, zxy: float) -> float:
    """Меньшая главная кривизна графика, выпуклый купол — плюс."""
    p2q2 = zx * zx + zy * zy
    w2 = 1.0 + p2q2
    w = w2 ** 0.5
    ll, mm, nn = zxx / w, zxy / w, zyy / w
    mean = ((1.0 + zx * zx) * nn - 2.0 * zx * zy * mm + (1.0 + zy * zy) * ll) / (2.0 * w2)
    gauss = (ll * nn - mm * mm) / w2
    rad = (max(0.0, mean * mean - gauss)) ** 0.5
    # Большая по алгебре главная кривизна ближе к нулю у вытянутого вала и положительна у ямы.
    return float(-(mean + rad))


def lesser_curvature(z: np.ndarray, row: int, col: int, res: float = 1.0, span_m: float = 3.0) -> float:
    """Меньшая главная кривизна, 1/м, по квадратичной регрессии диска.

    Восемь отсчётов на шуме лидара в несколько сантиметров дают ложную кривизну
    порядка σ√6 / R². Квадратичный полином по всем ячейкам диска этот шум усредняет.
    На вершине профильная кривизна не определена. У купола выпуклы обе главные,
    у вала одна: берётся меньшая. В канале curv знак обратный, канал не меняется.
    """
    if res <= 0 or span_m <= 0:
        return np.nan
    rad = max(1, int(np.ceil(float(span_m) / float(res) - 1e-9)))
    r0, r1 = max(0, row - rad), min(z.shape[0], row + rad + 1)
    c0, c1 = max(0, col - rad), min(z.shape[1], col + rad + 1)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    x = (xx - col) * float(res)
    y = (row - yy) * float(res)
    patch = z[r0:r1, c0:c1]
    reach = float(span_m) / float(res)
    mask = (x * x + y * y <= float(span_m) ** 2) & np.isfinite(patch)
    if int(mask.sum()) < max(12, 0.5 * np.pi * reach * reach):
        return np.nan
    x, y = x[mask].astype(np.float64), y[mask].astype(np.float64)
    zz = patch[mask].astype(np.float64)
    design = np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])
    coef, *_ = np.linalg.lstsq(design, zz, rcond=None)
    return _principal_min(float(coef[1]), float(coef[2]), 2.0 * float(coef[3]),
                          2.0 * float(coef[4]), float(coef[5]))


def priors(z: np.ndarray, res: float, xs: np.ndarray, ys: np.ndarray, left: float, top: float,
           near_m: float = 5.0, far_m: float = 40.0) -> np.ndarray:
    """Множитель уверенности по каждому центру. Ниже 1 не опускается."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    out = np.ones(xs.shape[0], np.float32)
    if xs.size == 0 or res <= 0:
        return out
    near_px = float(near_m) / float(res)
    far_px = max(near_px + 1.0, float(far_m) / float(res))
    for i, (x, y) in enumerate(zip(xs, ys)):
        # Ячейка, которая содержит точку: левый и верхний край, как у from_origin.
        col = int(np.floor((x - left) / res))
        row = int(np.floor((top - y) / res))
        near = tpi(z, row, col, near_px)
        far = tpi(z, row, col, far_px)
        sym = symmetry(z, row, col, near_px)
        curv = lesser_curvature(z, row, col, res, span_m=near_m)
        # Нет измерения — не голос против. Плоскость симметрична, но это не насыпь:
        # симметрия считается только вместе с превышением центра над кольцом.
        checks = []
        if np.isfinite(near):
            checks.append(bool(near > 0.05))
        if np.isfinite(near) and np.isfinite(far):
            checks.append(bool(near > far))
        if np.isfinite(sym) and np.isfinite(near):
            checks.append(bool(sym >= 0.65 and near > 0.05))
        if np.isfinite(curv):
            checks.append(bool(curv > 0.002))
        votes = sum(checks)
        known = len(checks)
        if known and votes:
            # Только вверх. Повреждённая насыпь часто несимметрична: штраф
            # за «не купол» бил бы по самому тяжёлому классу.
            out[i] = np.float32(1.0 + 0.1 * (votes / known))
    return out
