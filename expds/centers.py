"""Карта центров кургана вместо маски.

Метрика засчитывает полигон, если внутренняя точка лежит в эталоне и покрытие
меньшей фигуры не меньше 0,5. Маска ставит эту точку в центр пятна, и пятно
часто сдвинуто относительно насыпи. Здесь цель обучения — гауссиана в центре
каждого объекта, а на выходе — круг вокруг пика. Круг меньше типичной насыпи,
поэтому меньшей фигурой оказывается он сам: при центре внутри эталона покрытие
равно единице.

Сданные веса лидара этот режим не меняют: в их чекпойнте нет target="centers".
Чекпойнт, в котором поле есть, и на лидаре отдаёт круги.
"""
from __future__ import annotations

import numpy as np

from . import accel

#: Во сколько раз пик гауссианы весит больше фона в BCE.
CENTER_POS_WEIGHT = 20.0


def gaussians_from_binary(mask, sigma_px: float, by_radius: bool = False, spike: bool = True,
                          edge_ignore: bool = False):
    """Бинарные маски N×K×H×W → гауссианы в центрах связных компонент.

    Несколько насыпей в одном тайле дают несколько пиков. Центр масс всей маски
    лёг бы между ними, и метрика такой объект не засчитывает.
    by_radius: ширина пика равна радиусу компоненты / 3, а не фиксированным метрам.
    spike=False оставляет центр между клетками (DARK, Zhang et al., CVPR 2020).
    С шипом ближайшая клетка равна 1: так училась сданная карта центров, и
    фокальная потеря видит положительный пиксель по порогу 0,999.
    edge_ignore: насыпь, которую режет край тайла, пика не получает, а её область
    (с запасом в σ) получает −1 — «не учить». Центр масс обрезка лежит не в центре
    кургана, а если центр вне тайла, пика в окне нет вовсе. Потери пропускают −1.
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
            ignore = np.zeros((h, w), bool)
            for lab in range(1, int(nlab) + 1):
                ys, xs = np.nonzero(labels == lab)
                cy = float(ys.mean())
                cx = float(xs.mean())
                if by_radius:
                    sigma = max(1.0, float(np.sqrt(len(ys) / np.pi)) / 3.0)
                else:
                    sigma = float(sigma_px)
                if edge_ignore and (ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1):
                    grow = max(1, int(np.ceil(sigma)))
                    ignore |= ndimage.binary_dilation(labels == lab, iterations=grow)
                    continue
                sig2 = 2.0 * sigma * sigma
                np.maximum(heat, np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / sig2), out=heat)
                if spike:
                    ry = min(max(int(round(cy)), 0), h - 1)
                    rx = min(max(int(round(cx)), 0), w - 1)
                    heat[ry, rx] = 1.0
            if edge_ignore and ignore.any():
                heat[ignore & (heat < 0.5)] = -1.0
    return torch.from_numpy(out).to(device)


def small_component_weights(mask, weight: float, max_frac: float = 0.002, grow_px: int = 2):
    """Вес потери N×K×H×W: связная компонента мельче max_frac кадра (с запасом grow_px) получает weight.

    Цель карты центров порогуется (> 0,5), поэтому умножать саму маску бесполезно: вес
    мелкой насыпи должен войти в потерю. weight == 1 даёт None — потеря как раньше.
    """
    import torch
    from scipy import ndimage

    if float(weight) == 1.0:
        return None
    binary = mask.detach().cpu().numpy() > 0.5
    n, k, h, w = binary.shape
    out = np.ones((n, k, h, w), np.float32)
    limit = float(max_frac) * h * w
    for i in range(n):
        for c in range(k):
            if not binary[i, c].any():
                continue
            labels, nlab = ndimage.label(binary[i, c])
            sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, nlab + 1))
            small = np.flatnonzero(sizes < limit) + 1
            if small.size:
                zone = np.isin(labels, small)
                if grow_px > 0:
                    zone = ndimage.binary_dilation(zone, iterations=int(grow_px))
                out[i, c][zone] = float(weight)
    return torch.from_numpy(out).to(mask.device)


def large_component_weights(mask, class_weights, min_px: float, grow_px: int = 2):
    """Вес потери N×K×H×W для КРУПНЫХ насыпей (sp_large, 31.10): противоположность small_component_weights.

    class_weights — множитель по каналу класса (1.0 — канал не трогать). Компонента маски класса,
    площадь которой не меньше min_px пикселей (порог в м² эталона, пересчитанный вызывающим кодом
    через шаг и сжатие кэша), получает свой вес с запасом grow_px. Все веса 1 — None (потеря как раньше).
    """
    import torch
    from scipy import ndimage

    cw = [float(w) for w in class_weights]
    if all(w == 1.0 for w in cw):
        return None
    binary = mask.detach().cpu().numpy() > 0.5
    n, k, h, w = binary.shape
    out = np.ones((n, k, h, w), np.float32)
    for i in range(n):
        for c in range(k):
            if c >= len(cw) or cw[c] == 1.0 or not binary[i, c].any():
                continue
            labels, nlab = ndimage.label(binary[i, c])
            sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, nlab + 1))
            big = np.flatnonzero(sizes >= float(min_px)) + 1
            if big.size:
                zone = np.isin(labels, big)
                if grow_px > 0:
                    zone = ndimage.binary_dilation(zone, iterations=int(grow_px))
                out[i, c][zone] = cw[c]
    return torch.from_numpy(out).to(mask.device)


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


def _center_positives(target):
    """Пиксели, которые потеря считает пиками.

    Шип в единицу ловится порогом 0,999. Гауссиана без шипа не достигает
    единицы, если центр между клетками: тогда пик — локальный максимум выше 0,5.
    На цели с шипом это тот же пиксель, потеря сданного рецепта не меняется.
    """
    import torch.nn.functional as F

    pooled = F.max_pool2d(target, 3, stride=1, padding=1)
    return (target >= 0.999) | ((target + 1e-6 >= pooled) & (target > 0.5))


def focal_center_loss(logits, target, alpha: float = 2.0, beta: float = 4.0, weight=None):
    """Фокальная потеря карты центров (Law & Deng, 2018; Zhou et al., 2019).

    У пика (target = 1) штраф (1 − p)^α. Рядом с пиком фон почти не штрафуется:
    множитель (1 − target)^β. Нормировка — на число пиков в батче, не на все пиксели.
    weight (N×K×H×W) умножает штраф пика: мелкая насыпь весит больше (`small_component_weights`).
    """
    import torch

    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    valid = target >= 0
    target = target.clamp(min=0)
    pos = _center_positives(target)
    neg = ~pos & valid
    pos_loss = -((1 - p) ** alpha) * torch.log(p)
    if weight is not None:
        pos_loss = pos_loss * weight
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p)
    n_pos = pos.sum().clamp(min=1).to(pos_loss.dtype)
    return (pos_loss[pos].sum() + neg_loss[neg].sum()) / n_pos


def center_loss(logits, target, weight=None):
    """BCE с весом на пике плюс Dice только по классам, которые есть в батче.

    weight (N×K×H×W) дополнительно умножает вес BCE: мелкая насыпь весит больше.
    """
    import torch.nn.functional as F

    from .train import dice_loss

    valid = (target >= 0).float()
    target = target.clamp(min=0)
    w = (1.0 + CENTER_POS_WEIGHT * target) * valid
    if weight is not None:
        w = w * weight
    return (F.binary_cross_entropy_with_logits(logits.float(), target, weight=w)
            + dice_loss(logits.float() * valid - 20.0 * (1 - valid), target))


def heatmap_counts(pred, target, radius: int, thr: float = 0.5):
    """TP, FP, FN по классам: пик эталона найден, если рядом есть пик предсказания.

    pred и target — вероятности N×K×H×W. Пик — локальный максимум окна
    (2r+1) выше 0,5. Совпадение — в том же радиусе, в пикселях.
    """
    import torch.nn.functional as F

    r = max(int(radius), 1)
    k = 2 * r + 1

    def peaks(p):
        pooled = F.max_pool2d(p, k, stride=1, padding=r)
        return (p + 1e-6 >= pooled) & (p > th)

    def _gt_peaks(p):
        pooled = F.max_pool2d(p, k, stride=1, padding=r)
        return (p + 1e-6 >= pooled) & (p > 0.5)

    th = float(thr)
    pr = peaks(pred) & (target >= 0)
    gt = _gt_peaks(target.clamp(min=0))
    gt_near = F.max_pool2d(gt.float(), k, stride=1, padding=r) > 0
    pr_near = F.max_pool2d(pr.float(), k, stride=1, padding=r) > 0
    dims = (0, 2, 3)
    tp = (pr & gt_near).sum(dim=dims)
    fp = (pr & ~gt_near).sum(dim=dims)
    fn = (gt & ~pr_near).sum(dim=dims)
    return tp, fp, fn


def _log_parabola(left: float, mid: float, right: float) -> float:
    """Шаг Ньютона по логарифму трёх отсчётов: μ = m − P'(m) / P''(m).

    Это одномерный частный случай DARK (Zhang et al., CVPR 2020): логарифм
    гауссианы квадратичен, и разложение Тейлора в целом пике возвращает её центр.
    Положительный сдвиг — к большему индексу. Обрезка ±0,5 не даёт перескочить
    на соседнюю клетку. Перекрёстная производная не считается: цели обучения
    изотропны, у них она нулевая.
    """
    sample = np.array([left, mid, right], np.float64)
    if not np.isfinite(sample).all():
        return 0.0
    vals = np.log(np.clip(sample, 1e-6, None))
    denom = float(vals[0] - 2.0 * vals[1] + vals[2])
    if abs(denom) < 1e-8:
        return 0.0
    return float(np.clip(0.5 * (vals[0] - vals[2]) / denom, -0.5, 0.5))


def subpixel_offsets(heat: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сдвиг каждого пика внутри своей клетки. На краю растра сдвиг по этой оси нулевой."""
    height, width = heat.shape
    dy = np.zeros(len(rows), np.float64)
    dx = np.zeros(len(cols), np.float64)
    for i, (row, col) in enumerate(zip(rows.astype(int), cols.astype(int))):
        if 0 < row < height - 1:
            dy[i] = _log_parabola(heat[row - 1, col], heat[row, col], heat[row + 1, col])
        if 0 < col < width - 1:
            dx[i] = _log_parabola(heat[row, col - 1], heat[row, col], heat[row, col + 1])
    return dy, dx


def _keep_min_distance(xs: np.ndarray, ys: np.ndarray, min_dist: float) -> np.ndarray:
    """Оставляет пики не ближе min_dist. Вход уже отсортирован по убыванию уверенности."""
    keep = np.ones(len(xs), dtype=bool)
    if min_dist <= 0 or len(xs) < 2:
        return keep
    limit = float(min_dist) ** 2
    chosen: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if any((float(x) - cx) ** 2 + (float(y) - cy) ** 2 < limit for cx, cy in chosen):
            keep[i] = False
        else:
            chosen.append((float(x), float(y)))
    return keep


def centers_to_features(probs: np.ndarray, classes: list[str], transform, crs: str | None, res: float,
                        region: str, markup: str, source: str, radius_m: float = 6.0,
                        min_dist_m: float = 12.0, low: float = 0.05,
                        min_conf: dict[str, float] | None = None, subpixel: bool = False) -> list[dict]:
    """Пики карты центров → круги EPSG:3857. Уверенность — высота пика.

    radius_m меньше медианы насыпи на спутнике (~18 м), min_dist_m не даёт
    двум кругам слиться в один полигон. low низкий: отбор порогом делает
    подбор метрики, не этот шаг. Для OOF min_conf не задавать.
    subpixel сдвигает круг внутри клетки по параболе логарифма. По умолчанию
    выключен: пороги сдачи подобраны на целых пикселях.
    """
    from . import speed
    from .raster import circle_3857, circles_3857, pixel_to_xy

    out = []
    step = max(float(res), 1e-6)
    for k, cls in enumerate(classes):
        p = np.asarray(probs[k], np.float32)
        floor = float(min_conf[cls]) if min_conf and cls in min_conf else low
        rows, cols, vals = accel.find_peaks(p, max(1, int(round(min_dist_m / step))), floor)
        if rows.size == 0:
            continue
        if subpixel:
            dy, dx = subpixel_offsets(p, rows, cols)
            rows = rows.astype(np.float64) + dy
            cols = cols.astype(np.float64) + dx
        xs, ys = pixel_to_xy(transform, rows, cols)
        if subpixel:
            # Подавление в find_peaks шло по целым клеткам. Сдвиг может сблизить
            # два пика ближе min_dist_m, и круги склеятся в один полигон.
            keep = _keep_min_distance(xs, ys, min_dist_m)
            xs, ys, vals = xs[keep], ys[keep], vals[keep]
        # флаг batch_circles ([speed], по умолчанию ВЫКЛ.): все круги одним вызовом GEOS/PROJ, побитно те же
        geoms = (circles_3857(xs, ys, radius_m, crs) if speed.on("batch_circles")
                 else (circle_3857(float(x), float(y), radius_m, crs) for x, y in zip(xs, ys)))
        for geom, v in zip(geoms, vals):
            if geom.is_empty:
                continue
            out.append({"geom": geom, "class_name": cls, "confidence": float(v), "region_name": region,
                        "markup_type": markup, "source": source, "original_crs": crs})
    return out
