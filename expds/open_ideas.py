"""Приёмы, которые по умолчанию не входят в сдачу.

Каждая функция сама по себе. Пайплайн вызывает её только если в конфиге есть
соответствующий ключ, а обучение — только если передан ненулевой флаг.
Пороги, веса и архив от этого файла не меняются.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from shapely.affinity import rotate, scale
from shapely.geometry import Point, box


def lost_fraction(total: int, kept: int) -> float:
    """Доля объектов класса, которые не попали ни в один учебный тайл."""
    if total <= 0:
        return 0.0
    return max(0, total - kept) / float(total)


def too_many_lost(total: int, kept: int, limit: float = 0.05) -> bool:
    """Правда, если потерянных объектов больше предела. Ноль объектов — не провал."""
    return total > 0 and lost_fraction(total, kept) > limit


def split_clipped(geoms: list, window, frac: float) -> tuple[list, list]:
    """Объект, от которого в окне меньше frac его площади, уходит в «не учить».

    frac <= 0 оставляет все геометрии в первом списке. Точки и пустые полигоны
    не считаются обрезком.
    """
    if frac <= 0:
        return list(geoms), []
    teach, ignore = [], []
    for geom in geoms:
        area = float(getattr(geom, "area", 0.0) or 0.0)
        if geom.geom_type == "Point" or area <= 0:
            teach.append(geom)
            continue
        part = geom.intersection(window)
        if float(part.area) < frac * area:
            ignore.append(geom)
        else:
            teach.append(geom)
    return teach, ignore


def ring_contrast(field: np.ndarray, row: int, col: int, radius_px: float) -> float:
    """Модуль разности среднего в центре и в кольце 0,85–1,55 радиуса.

    Светлое и тёмное кольцо дают одно и то же число. Радиус меньше пикселя — ноль.
    """
    radius = float(radius_px)
    if radius < 1.0:
        return 0.0
    h, w = field.shape
    r0, r1 = max(0, int(row - 1.55 * radius)), min(h, int(row + 1.55 * radius) + 1)
    c0, c1 = max(0, int(col - 1.55 * radius)), min(w, int(col + 1.55 * radius) + 1)
    yy, xx = np.mgrid[r0:r1, c0:c1]
    dist = np.hypot(yy - row, xx - col)
    center = dist <= 0.5 * radius
    ring = (dist >= 0.85 * radius) & (dist <= 1.55 * radius)
    patch = np.asarray(field[r0:r1, c0:c1], np.float64)
    good = np.isfinite(patch)
    if not (center & good).any() or not (ring & good).any():
        return 0.0
    return float(abs(patch[center & good].mean() - patch[ring & good].mean()))


def repair_dead_pixels(image: np.ndarray) -> np.ndarray:
    """Ноль и выброс относительно соседей заменяются средним ненулевых соседей.

    image — H×W×C. Центр в среднее не входит. Канал без живых соседей не меняется.
    """
    src = np.asarray(image)
    if src.ndim != 3:
        raise ValueError("ожидается H×W×C")
    out = src.copy()
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    for channel in range(src.shape[2]):
        band = src[:, :, channel].astype(np.float64)
        local_min = ndi.minimum_filter(band, footprint=kernel, mode="reflect")
        local_max = ndi.maximum_filter(band, footprint=kernel, mode="reflect")
        mean = float(np.mean(band))
        dead = (band == 0) | (band < local_min - 0.5 * abs(mean)) | (band > local_max + abs(mean))
        neighbor_sum = ndi.convolve(np.where(band == 0, 0.0, band), kernel.astype(np.float64), mode="reflect")
        neighbor_n = ndi.convolve((band != 0).astype(np.float64), kernel.astype(np.float64), mode="reflect")
        fill = np.divide(neighbor_sum, neighbor_n, out=band.copy(), where=neighbor_n > 0)
        band = np.where(dead & (neighbor_n > 0), fill, band)
        out[:, :, channel] = band.astype(src.dtype, copy=False)
    return out


def component_from_peak(score: np.ndarray, row: int, col: int, rel: float = 0.75) -> np.ndarray:
    """Связная область вокруг пика: порог 0,75 его высоты, но не ниже 0,35 и не выше 0,8.

    score уже приведён к [0, 1]. Пик вне массива даёт пустую маску.
    """
    field = np.asarray(score, np.float64)
    out = np.zeros(field.shape, bool)
    if not (0 <= row < field.shape[0] and 0 <= col < field.shape[1]):
        return out
    height = float(field[row, col])
    if not np.isfinite(height):
        return out
    level = max(0.35, min(0.8, height * rel))
    labeled, n = ndi.label(np.isfinite(field) & (field >= level))
    mark = int(labeled[row, col])
    if mark == 0 or n == 0:
        return out
    return labeled == mark


def ellipse_of(poly, n: int = 32):
    """Эллипс по минимальному повёрнутому прямоугольнику. Точка возвращается как есть."""
    if poly.geom_type == "Point" or poly.is_empty:
        return poly
    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    edges = []
    for a, b in zip(coords, coords[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        edges.append((dx * dx + dy * dy, dx, dy))
    edges.sort(reverse=True)
    long = max(edges[0][0], 1e-12) ** 0.5
    short = max(edges[-1][0], 1e-12) ** 0.5
    angle = float(np.degrees(np.arctan2(edges[0][2], edges[0][1])))
    center = poly.centroid
    circle = Point(center.x, center.y).buffer(1.0, quad_segs=max(2, n // 4))
    return rotate(scale(circle, long / 2.0, short / 2.0, origin=center), angle, origin=center)


def medial_peaks(mask: np.ndarray, min_distance: int = 2) -> list[tuple[int, int]]:
    """Локальные максимумы дистанционного преобразования. Дыра маски пиком не становится."""
    binary = np.asarray(mask, bool)
    if not binary.any():
        return []
    dist = ndi.distance_transform_edt(binary)
    dilated = ndi.maximum_filter(dist, size=max(3, int(min_distance) * 2 + 1))
    peaks = (dist == dilated) & (dist > 0) & binary
    ys, xs = np.nonzero(peaks)
    return [(int(r), int(c)) for r, c in zip(ys, xs)]


def choose_stable_threshold(scored: list[tuple[float, float]]) -> float:
    """Из порогов, которые не хуже лучшего больше чем на 0,002, берёт более высокий.

    scored — пары (порог, балл). Пустой список даёт ошибку: выбирать не из чего.
    Один прогон часто берёт шумный пик; более высокий порог из той же полки
    режет лишние объекты, не отдавая заметный балл.
    """
    if not scored:
        raise ValueError("нет порогов")
    best = max(value for _, value in scored)
    shelf = [thr for thr, value in scored if value >= best - 0.002]
    return float(max(shelf))


def window_box(west: float, south: float, east: float, north: float):
    return box(west, south, east, north)


def mercator_scale(y: float) -> float:
    """Метров EPSG:3857 в метре на земле на широте точки y: 1 / cos(широты)."""
    lat = np.arctan(np.sinh(float(y) / 6378137.0))
    return float(1.0 / max(np.cos(lat), 1e-6))


def thin_peaks(peaks: list[tuple[int, int]], value: np.ndarray, min_px: float) -> list[tuple[int, int]]:
    """Пики по убыванию value; пик ближе min_px к уже взятому выбрасывается."""
    order = sorted(peaks, key=lambda rc: -float(value[rc]))
    kept: list[tuple[int, int]] = []
    for r, c in order:
        if all((r - r2) ** 2 + (c - c2) ** 2 >= min_px * min_px for r2, c2 in kept):
            kept.append((r, c))
    return kept


def medial_fort_circles(feats: list[dict], radius_m: float, spacing_m: float = 25.0) -> list[dict]:
    """Фортификацию заменяет кругами вдоль оси стены, не чаще одного на spacing_m. Остальное не трогает.

    Метрика сопоставляет один к одному: круги на одном эталонном вале — одно попадание и ложные.
    Выигрыш возможен, только если маска сливает несколько эталонных отрезков вала, поэтому шаг
    кругов — порядок длины отрезка, а не пиксель. Геометрии в EPSG:3857: шаг растра, радиус и
    шаг кругов переводятся из метров на земле множителем 1 / cos(широты).
    Короткий объект и слишком большая рамка остаются как были: ось там не надёжнее центра.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from scipy import ndimage as ndi_

    out = []
    for feat in feats:
        geom = feat.get("geom")
        if feat.get("class_name") != "fortifikatsii" or geom is None or geom.is_empty:
            out.append(feat)
            continue
        k = mercator_scale(geom.centroid.y) if abs(geom.centroid.y) > 1e4 else 1.0
        minx, miny, maxx, maxy = geom.bounds
        step = 1.0 * k
        width = int((maxx - minx) / step) + 2
        height = int((maxy - miny) / step) + 2
        if width < 3 or height < 3 or width * height > 2_000_000:
            out.append(feat)
            continue
        transform = from_origin(minx, maxy, step, step)
        mask = rasterize([(geom, 1)], out_shape=(height, width), transform=transform, dtype="uint8") > 0
        peaks = thin_peaks(medial_peaks(mask), ndi_.distance_transform_edt(mask), max(1.0, spacing_m))
        if len(peaks) < 2:
            out.append(feat)
            continue
        for row, col in peaks:
            x, y = minx + (col + 0.5) * step, maxy - (row + 0.5) * step
            out.append(dict(feat, geom=Point(x, y).buffer(radius_m * k, quad_segs=4)))
    return out
