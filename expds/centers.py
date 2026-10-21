"""Карта центров кургана вместо маски.

Метрика засчитывает полигон, если внутренняя точка лежит в эталоне и покрытие
меньшей фигуры не меньше 0,5. Маска ставит эту точку в центр пятна, и пятно
часто сдвинуто относительно насыпи. Здесь цель обучения — гауссиана в центре
каждого объекта, а на выходе — круг вокруг пика. Круг меньше типичной насыпи,
поэтому меньшей фигурой оказывается он сам: при центре внутри эталона покрытие
равно единице.

Лидарные веса этот режим не меняет: в чекпойнте нет target="centers" — инференс
по-прежнему строит полигоны из маски.
"""
from __future__ import annotations

import numpy as np

from . import accel

#: Во сколько раз пик гауссианы весит больше фона в BCE.
CENTER_POS_WEIGHT = 20.0


def gaussians_from_binary(mask, sigma_px: float, by_radius: bool = False):
    """Бинарные маски N×K×H×W → гауссианы в центрах связных компонент.

    Несколько насыпей в одном тайле дают несколько пиков. Центр масс всей маски
    лёг бы между ними, и метрика такой объект не засчитывает.
    by_radius: ширина пика равна радиусу компоненты / 3, а не фиксированным метрам.
    """
    import torch
    from scipy import ndimage

    if not by_radius and float(sigma_px) <= 0:
        raise ValueError("sigma_px должен быть положительным")
    device = mask.device
    binary = mask.detach().cpu().numpy() > 0.5
    n, k, h, w = binary.shape
    out = np.zeros((n, k, h, w), np.float32)
    if not binary.any():
        return torch.from_numpy(out).to(device)
    yy, xx = np.ogrid[:h, :w]
    for i in range(n):
        for c in range(k):
            if not binary[i, c].any():
                continue
            labels, nlab = ndimage.label(binary[i, c])
            heat = out[i, c]
            for lab in range(1, int(nlab) + 1):
                ys, xs = np.nonzero(labels == lab)
                cy = float(ys.mean())
                cx = float(xs.mean())
                if by_radius:
                    sigma = max(1.0, float(np.sqrt(len(ys) / np.pi)) / 3.0)
                else:
                    sigma = float(sigma_px)
                sig2 = 2.0 * sigma * sigma
                np.maximum(heat, np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / sig2), out=heat)
                ry = min(max(int(round(cy)), 0), h - 1)
                rx = min(max(int(round(cx)), 0), w - 1)
                heat[ry, rx] = 1.0
    return torch.from_numpy(out).to(device)


def init_focal_bias(model, prior: float = 0.01) -> None:
    """Стартовая вероятность пика около prior, а не 0,5.

    Иначе сумма штрафа по всему фону в первые эпохи заставляет сеть
    предсказывать пустую карту: так и случилось на первом прогоне
    (потеря 4000 → 40, F1 пиков остался 0). Приём из CornerNet.
    """
    import math

    import torch

    conv = model.segmentation_head[0]
    if conv.bias is None:
        return
    torch.nn.init.constant_(conv.bias, -math.log((1.0 - prior) / prior))


def focal_center_loss(logits, target, alpha: float = 2.0, beta: float = 4.0):
    """Фокальная потеря карты центров (Law & Deng, 2018; Zhou et al., 2019).

    У пика (target = 1) штраф (1 − p)^α. Рядом с пиком фон почти не штрафуется:
    множитель (1 − target)^β. Нормировка — на число пиков в батче, не на все пиксели.
    """
    import torch

    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = target >= 0.999
    neg = ~pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p)
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p)
    n_pos = pos.sum().clamp(min=1).to(pos_loss.dtype)
    return (pos_loss[pos].sum() + neg_loss[neg].sum()) / n_pos


def center_loss(logits, target):
    """BCE с весом на пике плюс Dice только по классам, которые есть в батче."""
    import torch.nn.functional as F

    from .train import dice_loss

    weight = 1.0 + CENTER_POS_WEIGHT * target
    return (F.binary_cross_entropy_with_logits(logits.float(), target, weight=weight)
            + dice_loss(logits.float(), target))


def heatmap_counts(pred, target, radius: int):
    """TP, FP, FN по классам: пик эталона найден, если рядом есть пик предсказания.

    pred и target — вероятности N×K×H×W. Пик — локальный максимум окна
    (2r+1) выше 0,5. Совпадение — в том же радиусе, в пикселях.
    """
    import torch
    import torch.nn.functional as F

    r = max(int(radius), 1)
    k = 2 * r + 1

    def peaks(p):
        pooled = F.max_pool2d(p, k, stride=1, padding=r)
        return (p + 1e-6 >= pooled) & (p > 0.5)

    gt = peaks(target)
    pr = peaks(pred)
    gt_near = F.max_pool2d(gt.float(), k, stride=1, padding=r) > 0
    pr_near = F.max_pool2d(pr.float(), k, stride=1, padding=r) > 0
    dims = (0, 2, 3)
    tp = (pr & gt_near).sum(dim=dims)
    fp = (pr & ~gt_near).sum(dim=dims)
    fn = (gt & ~pr_near).sum(dim=dims)
    return tp, fp, fn


def centers_to_features(probs: np.ndarray, classes: list[str], transform, crs: str | None, res: float,
                        region: str, markup: str, source: str, radius_m: float = 6.0,
                        min_dist_m: float = 12.0, low: float = 0.05,
                        min_conf: dict[str, float] | None = None) -> list[dict]:
    """Пики карты центров → круги EPSG:3857. Уверенность — высота пика.

    radius_m меньше медианы насыпи на спутнике (~18 м), min_dist_m не даёт
    двум кругам слиться в один полигон. low низкий: отбор порогом делает
    подбор метрики, не этот шаг. Для OOF min_conf не задавать.
    """
    from .raster import circle_3857, pixel_to_xy

    out = []
    step = max(float(res), 1e-6)
    for k, cls in enumerate(classes):
        p = np.asarray(probs[k], np.float32)
        floor = float(min_conf[cls]) if min_conf and cls in min_conf else low
        rows, cols, vals = accel.find_peaks(p, max(1, int(round(min_dist_m / step))), floor)
        if rows.size == 0:
            continue
        xs, ys = pixel_to_xy(transform, rows, cols)
        for x, y, v in zip(xs, ys, vals):
            geom = circle_3857(float(x), float(y), radius_m, crs)
            if geom.is_empty:
                continue
            out.append({"geom": geom, "class_name": cls, "confidence": float(v), "region_name": region,
                        "markup_type": markup, "source": source, "original_crs": crs})
    return out
