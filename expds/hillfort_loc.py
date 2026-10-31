"""Точка городища на целом спутниковом кадре.

Ствол заморожен. Токены кадра и его горизонтального отражения усредняются.
Линейные головы дают карту. Круг ставится в максимум сглаженной карты, а не
по контуру кадра. Отдельное правило для уже посчитанной вероятности сцены:
ниже порога городища спутника этого участка снимаются, выше — добавляется
круг в заданной точке.

В configs/default.toml секции нет. pipeline вызывает фазу только при
[hillfort_loc] enabled = true.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

PATCH = 16
GROUP = "sp_dino_gor"


def grid_shape(width: int, height: int, side: int, patch: int = PATCH) -> tuple[int, int]:
    """Высота и ширина сетки: длинная сторона около side, обе кратны patch."""
    scale = float(side) / max(int(width), int(height), 1)
    h = max(patch, int(round(height * scale / patch)) * patch)
    w = max(patch, int(round(width * scale / patch)) * patch)
    return h, w


def score_tokens(tokens: np.ndarray, head: dict) -> np.ndarray:
    """Вероятность каждого токена. head: mean, basis, loc_mean, loc_scale, coef, intercept."""
    centered = np.asarray(tokens, np.float32) - np.asarray(head["mean"], np.float32)
    latent = centered @ np.asarray(head["basis"], np.float32).T
    scale = np.asarray(head["loc_scale"], np.float32)
    latent = (latent - np.asarray(head["loc_mean"], np.float32)) / np.maximum(scale, 1e-6)
    logit = latent @ np.asarray(head["coef"], np.float32) + float(head["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


def average_maps(tokens: np.ndarray, heads: list[dict], height: int, width: int) -> np.ndarray:
    """Среднее карт нескольких линейных голов, форма height×width."""
    if not heads:
        raise ValueError("hillfort_loc: нужен хотя бы один набор коэффициентов")
    maps = [score_tokens(tokens, head).reshape(height, width) for head in heads]
    return np.mean(np.stack(maps), axis=0).astype(np.float32)


def smooth_peak(score: np.ndarray, sigma: float = 0.5) -> tuple[int, int, float]:
    """Строка, столбец и высота максимума после гауссова сглаживания."""
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(np.asarray(score, np.float32), float(sigma), mode="nearest")
    flat = int(np.argmax(smooth))
    row, col = divmod(flat, smooth.shape[1])
    return row, col, float(smooth[row, col])


def pooled_tokens(model, image: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Токены кадра, усреднённые с горизонтальным отражением. Класс и регистры отброшены."""
    import torch

    image = np.asarray(image)
    flipped = image[:, ::-1]
    views = np.stack([image, flipped], axis=0).astype(np.float32) / 255.0
    views = (views - mean.reshape(1, 1, 1, 3)) / np.maximum(std.reshape(1, 1, 1, 3), 1e-6)
    batch = torch.from_numpy(np.transpose(views, (0, 3, 1, 2)).copy())
    try:
        batch = batch.to(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        pass
    hidden = model(pixel_values=batch).last_hidden_state
    registers = int(getattr(getattr(model, "config", None), "num_register_tokens", 0) or 0)
    tokens = hidden[:, 1 + registers :].detach().float().cpu().numpy()
    gh, gw = image.shape[0] // PATCH, image.shape[1] // PATCH
    width = tokens.shape[-1]
    straight = tokens[0].reshape(gh, gw, width)
    mirror = tokens[1].reshape(gh, gw, width)[:, ::-1]
    pooled = ((straight + mirror) * 0.5).reshape(gh * gw, width)
    return pooled.astype(np.float32), gh, gw


def scene_probability(tokens: np.ndarray, head: dict | None) -> float:
    """Вероятность «на кадре есть городище» по среднему и разбросу токенов. Нет головы — NaN."""
    if not head:
        return float("nan")
    pooled = np.concatenate([tokens.mean(0), tokens.std(0)]).astype(np.float32)
    return float(score_tokens(pooled.reshape(1, -1), head)[0])


def circle_at(x: float, y: float, radius_m: float):
    """Круг в EPSG:3857. Радиус — метры на земле."""
    from shapely.geometry import Point

    from .post import _ground_scale

    return Point(float(x), float(y)).buffer(float(radius_m) / _ground_scale(float(y)), quad_segs=8)


def feature_at(x: float, y: float, confidence: float, region: str, radius_m: float = 2.0,
                source: str = "hillfort_loc") -> dict:
    return {
        "geom": circle_at(x, y, radius_m),
        "class_name": "gorodishcha",
        "confidence": float(confidence),
        "region_name": region,
        "markup_type": "sp_or",
        "threshold_group": GROUP,
        "source": source,
    }


def cell_center(transform, row: int, col: int, patch: int = PATCH) -> tuple[float, float]:
    """Центр клетки токена в координатах аффинного преобразования снимка."""
    return transform @ ((col + 0.5) * patch, (row + 0.5) * patch)


def apply_scene_gate(feats: list[dict], scenes: list[dict], thr: float = 0.1,
                      radius_m: float = 2.0) -> list[dict]:
    """scenes: region, p, x, y. Низкая p снимает городища спутника участка, высокая добавляет круг."""
    low = {s["region"] for s in scenes if float(s["p"]) < thr}
    out = []
    for feat in feats:
        satellite = feat.get("markup_type") == "sp_or" or feat.get("threshold_group") in ("sp_or", GROUP)
        if feat.get("region_name") in low and feat.get("class_name") == "gorodishcha" and satellite:
            continue
        out.append(feat)
    for scene in scenes:
        if float(scene["p"]) < thr:
            continue
        out.append(feature_at(scene["x"], scene["y"], scene["p"], scene["region"], radius_m, "scene_gate"))
    return out


def locate(tokens: np.ndarray, heads: list[dict], height: int, width: int, transform,
           region: str, radius_m: float = 2.0, sigma: float = 0.5,
           scene_head: dict | None = None, crs: str | None = None) -> dict:
    """Один круг по уже посчитанным токенам. transform — аффин сетки токенов."""
    from .crs import WEB_MERCATOR, to_crs
    from shapely.geometry import Point

    score = average_maps(tokens, heads, height, width)
    row, col, peak = smooth_peak(score, sigma)
    x, y = cell_center(transform, row, col)
    if crs and str(crs).upper().replace(" ", "") != "EPSG:3857":
        point = to_crs(Point(x, y), crs, WEB_MERCATOR)
        x, y = float(point.x), float(point.y)
    scene = scene_probability(tokens, scene_head)
    confidence = peak if math.isnan(scene) else 0.5 * (peak + scene)
    feat = feature_at(x, y, confidence, region, radius_m)
    feat["peak"] = peak
    feat["scene_p"] = scene
    return feat


def _head_from(packed, prefix: str) -> dict:
    head = {key: np.array(packed[f"{prefix}_{key}"]) for key in ("mean", "basis", "loc_mean", "loc_scale", "coef")}
    head["intercept"] = float(packed[f"{prefix}_intercept"])
    return head


def load_pack(path: str):
    """Коэффициенты нашего формата: n_heads и массивы h0_mean, h0_basis, …"""
    with np.load(path) as packed:
        heads = [_head_from(packed, f"h{i}") for i in range(int(packed["n_heads"]))]
        scene = _head_from(packed, "scene") if "scene_mean" in packed else None
        mean = np.asarray(packed["image_mean"], np.float32)
        std = np.asarray(packed["image_std"], np.float32)
    return heads, scene, mean, std


def read_rgb(path, side: int):
    """Кадр длинной стороной около side. Возвращает uint8 HWC, аффин этой сетки и CRS."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        height, width = grid_shape(ds.width, ds.height, side)
        count = min(3, ds.count)
        raw = ds.read(indexes=list(range(1, count + 1)), out_shape=(count, height, width),
                      resampling=Resampling.average).astype(np.float32)
        from rasterio.transform import Affine

        # Долгая сторона сжата: один пиксель сетки покрывает (ширина_снимка / ширина_сетки) пикселей файла.
        transform = ds.transform * Affine.scale(ds.width / width, ds.height / height)
        crs = ds.crs.to_string() if ds.crs is not None else None
    if raw.shape[0] == 1:
        raw = np.repeat(raw, 3, axis=0)
    if float(raw.max()) <= 1.5:
        raw = raw * 255.0
    elif float(raw.max()) > 255.0:
        lo, hi = np.percentile(raw, (1, 99))
        raw = (raw - lo) / max(float(hi - lo), 1e-6) * 255.0
    image = np.clip(raw, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return image, transform, crs


def phase(regions, cfg: dict) -> list[dict]:
    """Пусто, пока секция выключена или нет своих коэффициентов и ствола."""
    spec = cfg.get("hillfort_loc") or {}
    if not spec.get("enabled"):
        return []
    from pathlib import Path

    weights, backbone = spec.get("weights"), spec.get("backbone")
    if not weights or not backbone or not Path(weights).is_file():
        log.warning("hillfort_loc включён, но нет weights или backbone — фаза пустая")
        return []
    try:
        import torch
        from transformers import AutoModel
    except Exception:
        log.warning("hillfort_loc: нет torch или transformers — фаза пустая")
        return []
    heads, scene_head, mean, std = load_pack(weights)
    local = Path(backbone)
    model = AutoModel.from_pretrained(backbone, local_files_only=local.is_dir()).eval()
    side = int(spec.get("side", 512))
    radius = float(spec.get("radius_m", 2.0))
    found = []
    for reg in regions:
        for path in (getattr(reg, "rasters", {}) or {}).get("sp_or", []):
            try:
                image, transform, crs = read_rgb(path, side)
                with torch.inference_mode():
                    tokens, gh, gw = pooled_tokens(model, image, mean, std)
                found.append(locate(tokens, heads, gh, gw, transform, reg.name, radius,
                                    scene_head=scene_head, crs=crs))
            except Exception:
                log.exception("hillfort_loc: снимок пропущен %s", path)
    return found
