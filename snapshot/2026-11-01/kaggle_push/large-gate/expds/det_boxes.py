"""Рамки из маски тайла и круг в той же точке, что у карты центров.

Это подготовка сравнения детектора рамок с U-Net. Пакет детектора сюда не
ставится, сдача его не вызывает. Кэш, из которого берутся тайлы, собран вокруг
объектов со сдвигом, а инференс идёт сплошной сеткой: на таком кэше детектор
учит «объект в центре кадра». Сравнение с U-Net на инференсе этим экспортом
не заменяется. Приёмка — официальная метрика на честных фолдах, не AP рамок.

Окно в индексе задано центром (cx, cy), шагом и стороной. Настоящее окно ещё
привязано к сетке растра, а центр в индексе округлён до сантиметра. Обратный
пересчёт без привязки окна ошибается не больше чем на полпикселя и 5 мм.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .tiles import CLASS_BITS_MASK, IGNORE_BIT


def boxes_from_mask(mask: np.ndarray, classes: list[str], ignore_bit: int = IGNORE_BIT) -> list[dict]:
    """Связные области своих классов → рамки COCO [x, y, ширина, высота] в пикселях.

    Бит «не учить» рамки не даёт. Начало координат — левый верхний угол пикселя (0, 0).
    """
    if ignore_bit < 1:
        raise ValueError(f"бит «не учить» {ignore_bit}")
    own = np.asarray(mask) & ((1 << ignore_bit) - 1) & CLASS_BITS_MASK
    out = []
    for k, name in enumerate(classes):
        bit = ((own >> k) & 1).astype(bool)
        if not bit.any():
            continue
        labeled, n = ndi.label(bit)
        for i in range(1, int(n) + 1):
            ys, xs = np.nonzero(labeled == i)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            out.append({
                "category": name,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "centroid_px": [float(xs.mean()), float(ys.mean())],
                "area_px": int(xs.size),
            })
    return out


def window_origin(cx: float, cy: float, res: float, tile: int) -> tuple[float, float]:
    """Западный и северный край окна, если оно стоит центром на (cx, cy) без привязки к сетке."""
    half = tile * float(res) / 2.0
    return float(cx) - half, float(cy) + half


def box_center_world(bbox: list[float], cx: float, cy: float, res: float, tile: int) -> tuple[float, float]:
    """Центр рамки COCO в системе координат тайла. (cx, cy) — центр окна."""
    x, y, w, h = (float(v) for v in bbox)
    west, north = window_origin(cx, cy, res, tile)
    return west + (x + w / 2.0) * float(res), north - (y + h / 2.0) * float(res)


def circle_from_box(bbox: list[float], cx: float, cy: float, res: float, tile: int,
                    radius_m: float, crs: str | None, category: str, confidence: float) -> dict:
    """Круг того же радиуса, что у карты центров, в центре рамки. Уверенность не пересчитывается."""
    from .raster import circle_3857

    x, y = box_center_world(bbox, cx, cy, res, tile)
    return {
        "class_name": category,
        "confidence": float(confidence),
        "geometry": circle_3857(x, y, float(radius_m), crs),
    }
