"""Быстрые вычисления: Rust-модуль expds_core или эквивалент на NumPy.

Rust подключается автоматически, если модуль собран (maturin develop) или
лежит в inference/lib/. Принудительно выбрать NumPy: EXPDS_BACKEND=numpy.
Обе реализации дают одинаковый результат (см. tests/test_accel.py):
отмывки совпадают с точностью ±1 уровень яркости на единичных пикселях.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _load_rust():
    """Модуль expds_core, если он собран. Иначе None — те же функции идут на NumPy.

    Точка входа площадки кладёт inference/lib в путь сама. Скрипты и тесты
    импортируют expds напрямую, и без этой подстановки Rust молча не находится.
    """
    if os.environ.get("EXPDS_BACKEND", "auto").lower() == "numpy":
        return None
    try:
        import expds_core as rs  # type: ignore
        return rs
    except ImportError:
        pass
    lib = Path(__file__).resolve().parents[1] / "inference" / "lib"
    if not lib.is_dir():
        return None
    sys.path.insert(0, str(lib))
    try:
        import expds_core as rs  # type: ignore
        return rs
    except ImportError:
        return None


_rs = _load_rust()

HAVE_RUST = _rs is not None


def backend() -> str:
    """Текущая реализация: 'rust <версия>' или 'numpy'."""
    return f"rust {_rs.version()}" if _rs is not None else "numpy"


# ───────────────────────────── LAS → сетка ─────────────────────────────

def grid_spec(min_x: float, min_y: float, max_x: float, max_y: float,
              res: float, max_pixels: int) -> dict:
    """Параметры сетки DEM (как бейзлайн): left, top, res, width, height, factor."""
    import math

    def extent(step: float):
        left = math.floor(min_x / step) * step
        bottom = math.floor(min_y / step) * step
        width = int(math.ceil((max_x - left) / step)) + 1
        height = int(math.ceil((max_y - bottom) / step)) + 1
        return left, bottom, width, height

    left, bottom, width, height = extent(res)
    factor = 1
    if max_pixels and width * height > max_pixels:
        factor = int(math.ceil(math.sqrt(width * height / max_pixels)))
        res = res * factor
        left, bottom, width, height = extent(res)
    return {"left": left, "top": bottom + height * res, "res": res,
            "width": width, "height": height, "factor": factor}


def _las_grid_numpy(path: str, res: float, ground_class: int, max_pixels: int,
                    use_all: bool, keep_prob: float = 1.0) -> tuple[np.ndarray, dict]:
    import laspy

    with laspy.open(path) as f:
        hdr = f.header
        spec = grid_spec(hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1], res, max_pixels)
        total_points = hdr.point_count

    def accumulate(all_points: bool, hist: dict | None):
        n = spec["width"] * spec["height"]
        sums = np.zeros(n, np.float64)
        counts = np.zeros(n, np.int64)
        used = 0
        with laspy.open(path) as f:
            for pts in f.chunk_iterator(2_000_000):
                cls = np.asarray(pts.classification)
                if hist is not None:
                    u, c = np.unique(cls, return_counts=True)
                    for a, b in zip(u, c):
                        hist[int(a)] = hist.get(int(a), 0) + int(b)
                m = np.ones(cls.shape, bool) if all_points else cls == ground_class
                if not m.any():
                    continue
                x = np.asarray(pts.x, np.float64)[m]
                y = np.asarray(pts.y, np.float64)[m]
                z = np.asarray(pts.z, np.float64)[m]
                if keep_prob < 1.0:
                    keep = np.fromiter(
                        (_keep_xyz(float(a), float(b), float(c), keep_prob) for a, b, c in zip(x, y, z)),
                        dtype=bool, count=x.size)
                    x, y, z = x[keep], y[keep], z[keep]
                    if x.size == 0:
                        continue
                col = np.floor((x - spec["left"]) / spec["res"])
                row = np.floor((spec["top"] - y) / spec["res"])
                ok = (col >= 0) & (row >= 0) & (col < spec["width"]) & (row < spec["height"])
                idx = row[ok].astype(np.int64) * spec["width"] + col[ok].astype(np.int64)
                sums += np.bincount(idx, weights=z[ok], minlength=n)
                counts += np.bincount(idx, minlength=n)
                used += int(ok.sum())
        return sums, counts, used

    hist: dict = {}
    sums, counts, used = accumulate(use_all, hist)
    used_all = use_all
    if used == 0 and not use_all and keep_prob >= 1.0:
        sums, counts, used = accumulate(True, None)
        used_all = True
    dem = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    dem = dem.reshape(spec["height"], spec["width"]).astype(np.float32)
    info = dict(spec, points_used=used, points_total=int(total_points), used_all=used_all, classes=hist)
    return dem, info


def _keep_xyz(x: float, y: float, z: float, keep_prob: float) -> bool:
    """Запасной путь без Rust. Доля стабильна между запусками, с Rust не обязана совпасть."""
    if keep_prob >= 1.0:
        return True
    if keep_prob <= 0.0:
        return False
    import struct
    raw = struct.pack("<ddd", float(x), float(y), float(z))
    h = 0x9E3779B97F4A7C15
    for byte in raw:
        h = (h ^ byte) * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
    return (h >> 11) / float(1 << 53) < keep_prob


def las_grid(path: str | Path, res: float = 0.1, ground_class: int = 2,
             max_pixels: int = 60_000_000, use_all: bool = False,
             keep_prob: float = 1.0) -> tuple[np.ndarray, dict]:
    """LAS/LAZ → DEM float32 (NaN в пустых ячейках) и словарь параметров сетки.

    keep_prob < 1 оставляет эту долю точек земли. 1 — сетка как раньше.
    """
    if _rs is not None:
        try:
            # keep_prob=1 — прежняя сигнатура: старый .so на платформе её принимает.
            if keep_prob < 1.0:
                return _rs.las_grid(str(path), res, ground_class, max_pixels, use_all, keep_prob)
            return _rs.las_grid(str(path), res, ground_class, max_pixels, use_all)
        except Exception as exc:  # редкие варианты LAS: читаем через laspy
            log.warning("Rust не прочитал %s (%s), перехожу на laspy", path, exc)
    return _las_grid_numpy(str(path), res, ground_class, max_pixels, use_all, keep_prob)


# ───────────────────────────── заполнение ─────────────────────────────

def resize_linear(src: np.ndarray, height: int, width: int) -> np.ndarray:
    """Билинейное изменение размера с семантикой cv2.resize(INTER_LINEAR)."""
    src = np.ascontiguousarray(src, np.float32)
    if _rs is not None:
        return _rs.resize_linear(src, height, width)
    sh, sw = src.shape

    def axis(s: int, d: int):
        scale = 1.0 / (d / s)
        f = ((np.arange(d) + 0.5) * scale - 0.5).astype(np.float32)
        i0 = np.floor(f).astype(np.int64)
        frac = f - i0.astype(np.float32)
        low = i0 < 0
        i0[low], frac[low] = 0, 0
        high = i0 >= s - 1
        i0[high], frac[high] = s - 1, 0
        i1 = np.minimum(i0 + 1, s - 1)
        return i0, i1, frac.astype(np.float32)

    x0, x1, fx = axis(sw, width)
    y0, y1, fy = axis(sh, height)
    rows = src[:, x0] * (np.float32(1) - fx) + src[:, x1] * fx
    out = rows[y0] * (np.float32(1) - fy)[:, None] + rows[y1] * fy[:, None]
    return out.astype(np.float32)


def fill_pyramid(dem: np.ndarray, max_levels: int = 16) -> np.ndarray:
    """Заполняет все NaN многомасштабной пирамидой (как бейзлайн)."""
    dem = np.ascontiguousarray(dem, np.float32)
    if _rs is not None:
        return _rs.fill_pyramid(dem, max_levels)
    current = dem
    levels = []
    for _ in range(max_levels):
        if np.isfinite(current).all() or min(current.shape) < 2:
            break
        levels.append(current)
        ph, pw = current.shape[0] % 2, current.shape[1] % 2
        if ph or pw:
            current = np.pad(current, ((0, ph), (0, pw)), constant_values=np.nan)
        valid = np.isfinite(current)
        vals = np.where(valid, current, 0).astype(np.float32)
        h2, w2 = current.shape[0] // 2, current.shape[1] // 2
        cnt = valid.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        sm = vals.reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        current = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan).astype(np.float32)
    finite = np.isfinite(current)
    if not finite.all():
        mean = float(current[finite].astype(np.float64).mean()) if finite.any() else 0.0
        current = np.where(finite, current, mean).astype(np.float32)
    for level in reversed(levels):
        up = resize_linear(current, level.shape[0], level.shape[1])
        current = np.where(np.isfinite(level), level, up).astype(np.float32)
    return current


def fill_tin(dem: np.ndarray) -> np.ndarray | None:
    """Линейная интерполяция пустот по Делоне занятых ячеек (Rust); None — модуль не собран или старый."""
    if _rs is None or not hasattr(_rs, "fill_tin"):
        return None
    return _rs.fill_tin(np.ascontiguousarray(dem, np.float32))


def convex_hull_mask(valid: np.ndarray) -> np.ndarray:
    """Маска выпуклой оболочки занятых ячеек (границы включительно)."""
    valid = np.ascontiguousarray(valid, bool)
    if _rs is not None:
        return _rs.convex_hull_mask(valid)
    from shapely.geometry import MultiPoint

    h, w = valid.shape
    rows = np.flatnonzero(valid.any(axis=1))
    if rows.size == 0:
        return valid.copy()
    c0 = valid[rows].argmax(axis=1)
    c1 = w - 1 - valid[rows][:, ::-1].argmax(axis=1)
    pts = np.concatenate([np.stack([c0, rows], 1), np.stack([c1, rows], 1)]).astype(float)
    if len(np.unique(pts, axis=0)) < 3:
        return valid.copy()
    hull = MultiPoint([tuple(p) for p in pts]).convex_hull
    if hull.geom_type != "Polygon":
        return valid.copy()
    xs, ys = np.asarray(hull.exterior.coords).T
    mask = np.zeros((h, w), bool)
    for y in range(max(0, int(ys.min())), min(h - 1, int(ys.max())) + 1):
        xl, xr = np.inf, -np.inf
        for i in range(len(xs) - 1):
            ax, ay, bx, by = xs[i], ys[i], xs[i + 1], ys[i + 1]
            lo, hi = (ay, by) if ay <= by else (by, ay)
            if y < lo - 1e-9 or y > hi + 1e-9:
                continue
            if abs(by - ay) < 1e-12:
                xl, xr = min(xl, ax, bx), max(xr, ax, bx)
            else:
                x = ax + (y - ay) * (bx - ax) / (by - ay)
                xl, xr = min(xl, x), max(xr, x)
        if xl <= xr:
            a, b = max(0, int(np.ceil(xl - 1e-9))), min(w - 1, int(np.floor(xr + 1e-9)))
            if b >= a:
                mask[y, a:b + 1] = True
    return mask


# ───────────────────────────── отмывки ─────────────────────────────

def hillshade(filled: np.ndarray, res_x: float, res_y: float, azimuth: float = 315.0,
              altitude: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """Серая отмывка ESRI/ArcGIS (uint8), как у бейзлайна организаторов.

    Азимут 315° — свет с северо-запада, чтобы тень падала на юго-восток.
    Полуденное солнце с юга переворачивает чтение рельефа и сюда не ставится.
    Углы не зависят от широты памятника: и на 60° с. ш., и южнее сеть лидара
    видела именно эту отмывку. Другое солнце — другие веса.
    """
    import math

    filled = np.ascontiguousarray(filled, np.float32)
    if _rs is not None:
        return _rs.hillshade(filled, res_x, res_y, azimuth, altitude, z_factor)
    gy = -np.gradient(filled, res_y, axis=0) if filled.shape[0] > 1 else np.zeros_like(filled)
    gx = np.gradient(filled, res_x, axis=1) if filled.shape[1] > 1 else np.zeros_like(filled)
    slope = np.arctan(np.hypot(gx, gy) * z_factor)
    aspect = np.arctan2(-gx, -gy)
    zen = math.radians(90.0 - altitude)
    az = math.radians(azimuth)
    shade = 255.0 * (math.cos(zen) * np.cos(slope) + math.sin(zen) * np.sin(slope) * np.cos(az - aspect))
    return np.rint(np.clip(shade, 0.0, 255.0)).astype(np.uint8)


def color_relief(filled: np.ndarray, valid: np.ndarray, z_min: float, z_range: float,
                 cycles: float) -> np.ndarray:
    """Цветная отмывка «Hot to cold» (3×H×W, uint8); вне съёмки — 0."""
    filled = np.ascontiguousarray(filled, np.float32)
    valid = np.ascontiguousarray(valid, bool)
    if _rs is not None:
        return _rs.color_relief(filled, valid, float(z_min), float(z_range), float(cycles))
    t = (filled - z_min) / z_range
    hue = (0.75 - cycles * t) % 1.0
    h = (hue % 1.0) * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    v = np.full(t.shape, 209.0 / 255.0, np.float32)
    s = np.ones_like(t)
    p, q, tt = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, tt, v])
    g = np.select(conds, [tt, v, v, q, p, p])
    b = np.select(conds, [p, p, tt, v, v, q])
    rgb = np.clip(np.rint(np.stack([r, g, b]).astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    rgb[:, ~valid] = 0
    return rgb


# ───────────────────────────── рельеф и пики ─────────────────────────────

def robust_z(a: np.ndarray) -> np.ndarray:
    """(a − медиана) / (1,4826·MAD) по конечным значениям; NaN остаются NaN."""
    a = np.ascontiguousarray(a, np.float32)
    if _rs is not None and hasattr(_rs, "robust_z"):
        return _rs.robust_z(a)
    fin = np.isfinite(a)
    if not fin.any():
        return np.full_like(a, np.nan)
    med = float(np.median(a[fin]))
    mad = float(np.median(np.abs(a[fin] - med))) * 1.4826 or 1.0
    return ((a - med) / mad).astype(np.float32)


def label_max(labels: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """out[l] = max(0, max values при labels == l), l = 0..n — как np.maximum.at; values конечные."""
    labels = np.ascontiguousarray(labels, np.int32)
    values = np.ascontiguousarray(values, np.float32)
    if _rs is not None and hasattr(_rs, "label_max") and labels.ndim == 2:
        return _rs.label_max(labels, values, int(n))
    out = np.zeros(n + 1, np.float32)
    np.maximum.at(out, labels.ravel(), values.ravel())
    return out


def box_mean(z: np.ndarray, radius: int) -> np.ndarray:
    """Среднее в окне (2r+1)² без учёта NaN (на краях — по попавшим пикселям)."""
    z = np.ascontiguousarray(z, np.float32)
    if _rs is not None:
        return _rs.box_mean(z, int(radius))
    fin = np.isfinite(z)
    s = np.pad(np.where(fin, z, 0).astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    n = np.pad(fin.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    del fin
    h, w = z.shape
    r0 = np.clip(np.arange(h) - radius, 0, h)
    r1 = np.clip(np.arange(h) + radius + 1, 0, h)
    c0 = np.clip(np.arange(w) - radius, 0, w)
    c1 = np.clip(np.arange(w) + radius + 1, 0, w)
    out = np.empty((h, w), np.float32)
    # W4: по блокам строк — те же поэлементные операции (результат побайтно тот же), пик памяти ниже
    # (участок 200, растр 14084×9034: полные float64-промежуточные не помещались в 7,5 ГБ)
    step = max(1, (1 << 22) // max(w, 1))
    for a in range(0, h, step):
        b = min(h, a + step)
        q0, q1 = r0[a:b], r1[a:b]
        tot = s[q1][:, c1] - s[q0][:, c1] - s[q1][:, c0] + s[q0][:, c0]
        cnt = n[q1][:, c1] - n[q0][:, c1] - n[q1][:, c0] + n[q0][:, c0]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(cnt > 0, tot / cnt, np.nan)
    return out


def slope_deg(z: np.ndarray, res: float) -> np.ndarray:
    """Уклон в градусах, ядро Хорна 3×3; край растра повторяется."""
    z = np.ascontiguousarray(z, np.float32)
    if _rs is not None and hasattr(_rs, "slope_deg"):
        return _rs.slope_deg(z, float(res))
    # Суммы ядра в float32 (как у Rust), деление и углы — в float64.
    p = np.pad(z, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]) - (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    dzdx = gx.astype(np.float64) / (8 * res)
    dzdy = gy.astype(np.float64) / (8 * res)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def ray_offsets(radius_px: int, n_dir: int) -> list[list[tuple[int, int, float]]]:
    """Смещения луча по каждому направлению: (dy, dx, расстояние), без повторов (как в Rust)."""
    import math

    rays = []
    for d in range(n_dir):
        a = 2 * math.pi * d / n_dir
        out: list[tuple[int, int, float]] = []
        for step in range(1, radius_px + 1):
            dx = round_half_away(step * math.cos(a))
            dy = -round_half_away(step * math.sin(a))
            if (dx, dy) == (0, 0) or (out and out[-1][:2] == (dy, dx)):
                continue
            out.append((dy, dx, math.hypot(dx, dy)))
        rays.append(out)
    return rays


def round_half_away(v: float) -> int:
    """Округление половинок от нуля (как f64::round в Rust), а не к чётному."""
    import math

    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def openness_svf(z: np.ndarray, res: float, radius_px: int, n_dir: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Положительная открытость (градусы) и фактор вида на небо по n_dir лучам; NaN вне съёмки.

    γ — наибольший угол возвышения горизонта по лучу (0, если на луче нет данных).
    Открытость — среднее (90° − γ), Yokoyama et al. (2002); SVF = 1 − среднее max(0, sin γ),
    Zakšek et al. (2011).
    """
    z = np.ascontiguousarray(z, np.float32)
    if _rs is not None and hasattr(_rs, "openness_svf"):
        return _rs.openness_svf(z, float(res), int(radius_px), int(n_dir))
    h, w = z.shape
    fin = np.isfinite(z)
    sum_open = np.zeros((h, w), np.float64)
    sum_sin = np.zeros((h, w), np.float64)
    z64 = z.astype(np.float64)
    for ray in ray_offsets(radius_px, n_dir):
        best = np.full((h, w), -np.inf)
        alive = np.ones((h, w), bool)
        for dy, dx, dist in ray:
            ys0, ys1 = max(0, -dy), min(h, h - dy)
            xs0, xs1 = max(0, -dx), min(w, w - dx)
            inside = np.zeros((h, w), bool)
            inside[ys0:ys1, xs0:xs1] = True
            alive &= inside
            nb = np.full((h, w), np.nan)
            nb[ys0:ys1, xs0:xs1] = z64[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            ok = alive & np.isfinite(nb)
            best = np.where(ok, np.maximum(best, (nb - z64) / (dist * res)), best)
        g = np.where(np.isfinite(best), np.arctan(best), 0.0)
        sum_open += 90.0 - np.degrees(g)
        sum_sin += np.maximum(np.sin(g), 0.0)
    n = max(1, n_dir)
    opn = np.where(fin, sum_open / n, np.nan).astype(np.float32)
    svf = np.where(fin, 1.0 - sum_sin / n, np.nan).astype(np.float32)
    return opn, svf


def local_relief(z: np.ndarray, radius: int) -> np.ndarray:
    """Локальный рельеф: высота минус среднее окрестности (пиксели)."""
    z = np.ascontiguousarray(z, np.float32)
    if _rs is not None:
        return _rs.local_relief(z, int(radius))
    return (z - box_mean(z, radius)).astype(np.float32)


def local_dev(z: np.ndarray, radius: int, std_floor: float = 0.05) -> np.ndarray:
    """(высота − среднее окна) / σ того же окна.

    Перед квадратом высота сдвигается на своё среднее: иначе z² абсолютной
    отметки съедает в float32 разницу в дециметры. На ровном поле σ не
    опускается ниже std_floor, иначе шум лидара становится бесконечным сигналом.
    """
    z64 = np.ascontiguousarray(z, np.float64)
    fin = np.isfinite(z64)
    ref = float(z64[fin].mean()) if fin.any() else 0.0
    shifted = np.where(fin, z64 - ref, np.nan).astype(np.float32)
    mean = box_mean(shifted, int(radius))
    sq = np.where(np.isfinite(shifted), shifted * shifted, np.nan).astype(np.float32)
    var = np.maximum(box_mean(sq, int(radius)) - mean * mean, 0.0)
    std = np.maximum(np.sqrt(var), np.float32(std_floor))
    return ((shifted - mean) / std).astype(np.float32)


def find_peaks(img: np.ndarray, radius: int, min_value: float,
               max_peaks: int = 1_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Локальные максимумы окна (2r+1)² не ниже min_value с подавлением ближе r пикселей."""
    img = np.ascontiguousarray(img, np.float32)
    if _rs is not None:
        return _rs.find_peaks(img, int(radius), float(min_value), int(max_peaks))
    from scipy.ndimage import maximum_filter

    clean = np.where(np.isfinite(img), img, -np.inf).astype(np.float32)
    mf = maximum_filter(clean, size=2 * radius + 1, mode="nearest")
    rows, cols = np.nonzero(np.isfinite(img) & (img >= np.float32(min_value)) & (img == mf))
    vals = img[rows, cols]
    order = np.lexsort((cols, rows, -vals))
    cell = max(radius, 1)
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
    out_r, out_c, out_v = [], [], []
    for k in order:
        if len(out_r) >= max_peaks:
            break
        r, c = int(rows[k]), int(cols[k])
        gr, gc = r // cell, c // cell
        close = any((qr - r) ** 2 + (qc - c) ** 2 <= radius * radius
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    for qr, qc in grid.get((gr + dr, gc + dc), ()))
        if not close:
            grid.setdefault((gr, gc), []).append((r, c))
            out_r.append(r)
            out_c.append(c)
            out_v.append(vals[k])
    return (np.asarray(out_r, np.int64), np.asarray(out_c, np.int64), np.asarray(out_v, np.float32))


def ring_stats(z: np.ndarray, rows: np.ndarray, cols: np.ndarray, r_in: float,
               r_out: float) -> np.ndarray:
    """Признаки кандидатов (N×6): диск, кольцо, std кольца, выраженность, min и max кольца."""
    z = np.ascontiguousarray(z, np.float32)
    rows = np.ascontiguousarray(rows, np.int64)
    cols = np.ascontiguousarray(cols, np.int64)
    if _rs is not None:
        return _rs.ring_stats(z, rows, cols, float(r_in), float(r_out))
    ro = int(np.ceil(r_out))
    dy, dx = np.mgrid[-ro:ro + 1, -ro:ro + 1]
    d = np.sqrt(dy * dy + dx * dx)
    out = np.full((len(rows), 6), np.nan, np.float32)
    h, w = z.shape
    for k, (r, c) in enumerate(zip(rows, cols)):
        y, x = r + dy, c + dx
        ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        v = np.full(d.shape, np.nan)
        v[ok] = z[y[ok], x[ok]]
        fin = np.isfinite(v)
        disk = v[fin & (d <= r_in)]
        ring = v[fin & (d > r_in) & (d <= r_out)]
        dm = disk.mean() if disk.size else np.nan
        if ring.size == 0:
            out[k, 0] = dm
            continue
        out[k] = [dm, ring.mean(), ring.std(), dm - ring.mean(), ring.min(), ring.max()]
    return out
