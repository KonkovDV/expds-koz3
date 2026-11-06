"""Обучение U-Net на кэше тайлов — режим для видеокарты 4 ГБ (GTX 1650).

Настройки по умолчанию: ResNet18 с весами ImageNet, тайл 256, батч 16,
fp16 (AMP), без тяжёлых аугментаций. Для отмывок лидара повороты и
отражения выключены: они меняют направление «освещения», и модель учит
артефакт. Проверка — по «семьям» участков (--val-families), а не случайно.

    python -m expds.train --cache work/cache/li --out models/li_r18.pt --val-families Гнёздово
    python -m expds.train --cache work/cache/spor --out models/spor_center.pt --target centers
    python -m expds.train --cache work/cache/li --out models/li_r18.pt --epochs 1 --limit 200   # дымовой тест
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import math
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Классы, чьи пиксели Simple Copy-Paste переносит на другой тайл.
PASTE_CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye")
#: Освещение одного снимка: яркость и контраст в долях диапазона [0, 1].
PHOTO_BRIGHTNESS = 0.08
PHOTO_CONTRAST = 0.15


def load_index(cache: Path) -> list[dict]:
    """Строки index.csv кэша."""
    with (cache / "index.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paste_channels(classes: list[str]) -> tuple[int, ...]:
    """Индексы классов, которые copy-paste переносит (курганы)."""
    return tuple(i for i, name in enumerate(classes) if name in PASTE_CLASSES)


def paste_objects(x, y, channels: tuple[int, ...], p: float,
                  blend: float = 0.0, scale: tuple[float, float] = (1.0, 1.0)):
    """Simple Copy-Paste: пиксели объекта с тайла-донора заменяют фон приёмника.

    x — N×C×H×W, y — N×K×H×W. Объект — пиксели каналов `channels`. Вставка только
    на тайл без кургана (с вероятностью p): плотная группа эталона не затирается,
    а пустой фон, где модель даёт ложные насыпи, получает настоящую форму.
    Донор — случайный тайл батча, где объект есть. Насыпь сдвигается так, чтобы её
    центр масс попал в случайный пиксель. Фон донора не копируется. Ghiasi et al., CVPR 2021.
    blend > 0 размывает край вставки окном усреднения. scale — случайный множитель
    размера вставленного объекта. Оба значения по умолчанию оставляют жёсткую вставку.
    """
    import torch
    import torch.nn.functional as F

    if p <= 0 or not channels or x.shape[0] == 0:
        return x, y
    n, _, h, w = x.shape
    obj = y[:, list(channels)].sum(dim=1) > 0
    has = obj.flatten(1).any(dim=1)
    donors = torch.where(has)[0]
    if donors.numel() == 0:
        return x, y
    do = (~has) & (torch.rand(n, device=x.device) < p)
    if not bool(do.any()):
        return x, y
    src = donors[torch.randint(donors.numel(), (n,), device=x.device)]
    taken = obj[src]
    mass = taken.flatten(1).sum(dim=1).clamp(min=1).to(x.dtype)
    rows = torch.arange(h, device=x.device, dtype=x.dtype).view(1, h, 1)
    cols = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, w)
    cy = (taken.to(x.dtype) * rows).sum(dim=(1, 2)) / mass
    cx = (taken.to(x.dtype) * cols).sum(dim=(1, 2)) / mass
    ty = torch.randint(0, h, (n,), device=x.device)
    tx = torch.randint(0, w, (n,), device=x.device)
    shift_y = (ty - cy.round()).view(n, 1, 1).to(torch.long)
    shift_x = (tx - cx.round()).view(n, 1, 1).to(torch.long)
    gy = torch.arange(h, device=x.device).view(1, h, 1).expand(n, h, w) - shift_y
    gx = torch.arange(w, device=x.device).view(1, 1, w).expand(n, h, w) - shift_x
    grid = torch.stack([
        gx.to(x.dtype) / max(w - 1, 1) * 2 - 1,
        gy.to(x.dtype) / max(h - 1, 1) * 2 - 1,
    ], dim=-1)
    sx = F.grid_sample(x[src], grid, mode="nearest", padding_mode="zeros", align_corners=True)
    sy = F.grid_sample(y[src], grid, mode="nearest", padding_mode="zeros", align_corners=True)
    alpha = (sy[:, list(channels)].sum(dim=1, keepdim=True) > 0.5) & do.view(n, 1, 1, 1)
    if blend <= 0 and scale[0] == 1.0 and scale[1] == 1.0:
        return torch.where(alpha, sx, x), torch.where(alpha, sy, y)
    if scale[0] != scale[1] or scale[0] != 1.0:
        factor = torch.empty(n, 1, 1, 1, device=x.device).uniform_(scale[0], scale[1])
        zoom = torch.zeros(n, h, w, 2, device=x.device)
        gy = torch.linspace(-1, 1, h, device=x.device).view(1, h, 1)
        gx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, w)
        zoom[..., 0] = gx / factor
        zoom[..., 1] = gy / factor
        sx = F.grid_sample(sx, zoom, mode="bilinear", padding_mode="zeros", align_corners=True)
        sy = F.grid_sample(sy, zoom, mode="nearest", padding_mode="zeros", align_corners=True)
        alpha = (sy[:, list(channels)].sum(dim=1, keepdim=True) > 0.5) & do.view(n, 1, 1, 1)
    a = alpha.to(x.dtype)
    if blend > 0:
        k = int(blend) * 2 + 1
        a = F.avg_pool2d(a, k, stride=1, padding=k // 2)
    return a * sx + (1.0 - a) * x, torch.where(a > 0.5, sy, y)


#: Значение выключенного канала до нормировки (x − 0,5) / 0,25: середина шкалы, то есть 0 после
#: нормировки. Чёрный 0 стал бы −2σ — сигналом, которого в данных нет.
CHANNEL_DROP_FILL = 0.5


def drop_channel(x, p: float):
    """С вероятностью p заменяет один канал кадра серединой шкалы. p <= 0 не меняет батч."""
    import torch

    if p <= 0 or x.shape[1] == 0:
        return x
    take = torch.rand(x.shape[0], device=x.device) < p
    if not bool(take.any()):
        return x
    which = torch.randint(x.shape[1], (x.shape[0],), device=x.device)
    out = x.clone()
    for i in torch.where(take)[0].tolist():
        out[i, int(which[i])] = CHANNEL_DROP_FILL
    return out


def elastic_deform(x, y, sigma: float, cell_px: int = 32):
    """Упругая деформация (Simard et al., 2003): гладкое поле сдвига, sigma — его амплитуда
    в долях полукадра (0,02 на тайле 256 — около 2,5 пикселя). sigma <= 0 возвращает батч как есть.

    Шум задаётся на грубой сетке с шагом cell_px и интерполируется: соседние пиксели сдвигаются
    вместе. Независимый шум в каждом пикселе рвал бы маску на крошки, и карта центров получила
    бы пик на каждой крошке.
    """
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return x, y
    n, _, h, w = x.shape
    gh, gw = max(2, h // max(1, int(cell_px)) + 1), max(2, w // max(1, int(cell_px)) + 1)
    coarse = torch.randn(n, 2, gh, gw, device=x.device) * float(sigma)
    noise = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
    base_y = torch.linspace(-1, 1, h, device=x.device).view(1, h, 1)
    base_x = torch.linspace(-1, 1, w, device=x.device).view(1, 1, w)
    grid = torch.stack([
        base_x.expand(n, h, w) + noise[..., 0],
        base_y.expand(n, h, w) + noise[..., 1],
    ], dim=-1).clamp(-1, 1)
    return (F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True),
            F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=True))


def emphasize_small(y, weight: float, max_frac: float = 0.002):
    """Умножает положительные пиксели кадра, если их доля меньше max_frac. weight == 1 не меняет y.

    Для карты центров не годится: цель порогуется по 0,5, и умноженная маска даёт ту же цель.
    Обучение берёт вес мелких насыпей в потерю (`centers.small_component_weights`).
    """
    import torch

    if weight == 1.0:
        return y
    flat = (y > 0.5).flatten(2).sum(dim=2)
    small = (flat > 0) & (flat < max_frac * y.shape[-1] * y.shape[-2])
    if not bool(small.any()):
        return y
    factor = torch.ones(y.shape[0], y.shape[1], 1, 1, device=y.device, dtype=y.dtype)
    factor[small] = float(weight)
    return y * factor


def photometric(x, brightness: float = PHOTO_BRIGHTNESS, contrast: float = PHOTO_CONTRAST):
    """Яркость и контраст кадра. x — N×C×H×W в [0, 1], маска не меняется.

    Контраст считается вокруг середины диапазона. Нулевые амплитуды оставляют
    снимок как есть. На проверке функцию не вызывают.
    """
    import torch

    if x.shape[0] == 0 or (brightness <= 0 and contrast <= 0):
        return x
    n = x.shape[0]
    c = 1 + (torch.rand(n, 1, 1, 1, device=x.device, dtype=x.dtype) * 2 - 1) * contrast
    b = (torch.rand(n, 1, 1, 1, device=x.device, dtype=x.dtype) * 2 - 1) * brightness
    return ((x - 0.5) * c + 0.5 + b).clamp(0, 1)


def resample_scale(x, y, amount: float, scales=None):
    """Случайный масштаб вокруг центра тайла, размер кадра тот же.

    amount=0.15 даёт масштаб 0.85–1.15. Масштаб меньше 1 увеличивает объект,
    больше 1 уменьшает. Снимок пересчитывается билинейно с повторением края,
    маска — ближайшим соседом, пустота снаружи остаётся пустой.
    """
    import torch
    import torch.nn.functional as F

    if amount <= 0 or x.shape[0] == 0:
        return x, y
    n = x.shape[0]
    if scales is None:
        scales = 1 + (torch.rand(n, device=x.device, dtype=x.dtype) * 2 - 1) * amount
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = scales
    theta[:, 1, 1] = scales
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    xs = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ys = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return xs, (ys > 0.5).to(y.dtype)


#: Каналы рельефа лидара, у которых высота входит линейно (локальный рельеф, м / клип).
RELIEF_LINEAR_TAGS = ("lrm_s", "lrm_l", "lrm_xl")


def _log_uniform(n: int, lo: float, hi: float, device, dtype):
    import torch

    if lo <= 0 or hi <= 0:
        raise ValueError("диапазон масштаба должен быть > 0")
    a, b = math.log(min(lo, hi)), math.log(max(lo, hi))
    return torch.exp(torch.empty(n, device=device, dtype=dtype).uniform_(a, b))


def relief_aug(x, y, li_tags, height=(1.0, 1.0), zoom=(1.0, 1.0), p: float = 0.0,
               slope_max_deg: float = 45.0):
    """Высота × h и масштаб × s кадра лидара с пересчётом каналов рельефа (идея, своя реализация).

    x — N×C×H×W в [0, 1] (до нормировки), y — N×K×H×W. Каналы по `li_tags` кэша:
      * lrm_* = clip(lr / clip_m, −1, 1)·0,5 + 0,5, lr линейна по высоте → u' = (u − 0,5)·h + 0,5
        (точно, кроме уже насыщенных пикселей);
      * slope = deg / slope_max: tan(угла) линеен по высоте и обратен масштабу → atan(h·tan / s);
      * прочие каналы только пересэмплируются.
    zoom s > 1 — объекты крупнее (кадр — центральная часть, растянутая в s раз),
    s < 1 — мельче (край повторяется). Окна lrm заданы в метрах и при масштабе не пересчитываются:
    при s в 0,7–1,45 это приближение (точный пересчёт — из DEM в кэше, v2).
    Каждый кадр с вероятностью p. p <= 0 — батч не меняется.
    """
    import torch
    import torch.nn.functional as F

    n = x.shape[0]
    if p <= 0 or n == 0 or (height == (1.0, 1.0) and zoom == (1.0, 1.0)):
        return x, y
    tags = list(li_tags or [])
    on = torch.rand(n, device=x.device) < p
    h = torch.where(on, _log_uniform(n, *height, x.device, x.dtype), torch.ones(n, device=x.device, dtype=x.dtype))
    s = torch.where(on, _log_uniform(n, *zoom, x.device, x.dtype), torch.ones(n, device=x.device, dtype=x.dtype))
    x = x.clone()
    for c, t in enumerate(tags[: x.shape[1]]):
        if t in RELIEF_LINEAR_TAGS:
            x[:, c] = ((x[:, c] - 0.5) * h.view(n, 1, 1) + 0.5).clamp(0, 1)
        elif t == "slope":
            ang = (x[:, c] * slope_max_deg).clamp(0, 89.0) * (math.pi / 180.0)
            ang = torch.atan(torch.tan(ang) * (h / s).view(n, 1, 1))
            x[:, c] = (ang * (180.0 / math.pi) / slope_max_deg).clamp(0, 1)
    if zoom != (1.0, 1.0):
        theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1.0 / s
        theta[:, 1, 1] = 1.0 / s
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        y = (F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5).to(y.dtype)
    return x, y


def site_weights(rows: list[dict], alpha: float, key: str = "region") -> np.ndarray:
    """Вес тайла ∝ n_участка^(−alpha): участки с тысячами тайлов не забивают эпоху.

    alpha = 0 — равные веса (как без балансировки), 1 — каждый участок поровну. Сумма = 1.
    """
    from collections import Counter

    cnt = Counter(r.get(key, "") for r in rows)
    w = np.array([cnt[r.get(key, "")] ** (-float(alpha)) for r in rows], dtype=np.float64)
    return w / max(w.sum(), 1e-12)


class TileDataset:
    """Тайлы из .npy-шардов (memmap); маски — биты классов → K каналов float."""

    def __init__(self, cache: Path, rows: list[dict], n_classes: int, flips: bool,
                 class_bits: list[int] | None = None, photo: bool = False, scale: float = 0.0,
                 channel_dropout: float = 0.0, elastic: float = 0.0):
        self.cache, self.rows, self.k, self.flips = cache, rows, n_classes, flips
        self.photo, self.scale, self.channel_dropout, self.elastic = photo, scale, channel_dropout, elastic
        self.bits = class_bits if class_bits is not None else list(range(n_classes))
        self._img: dict[int, np.ndarray] = {}
        self._msk: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _arrays(self, shard: int):
        if shard not in self._img:
            self._img[shard] = np.load(self.cache / f"images_{shard:04d}.npy", mmap_mode="r")
            self._msk[shard] = np.load(self.cache / f"masks_{shard:04d}.npy", mmap_mode="r")
        return self._img[shard], self._msk[shard]

    def __getitem__(self, i: int):
        import torch

        r = self.rows[i]
        imgs, msks = self._arrays(int(r["shard"]))
        x = np.asarray(imgs[int(r["idx"])], np.float32) / 255.0
        bits = np.asarray(msks[int(r["idx"])])
        y = np.stack([(bits >> k) & 1 for k in self.bits]).astype(np.float32)
        if self.flips:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        if self.scale > 0 or self.photo:
            import torch

            xt = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
            yt = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
            if self.scale > 0:
                xt, yt = resample_scale(xt, yt, self.scale)
            if self.photo:
                xt = photometric(xt)
            x, y = xt.squeeze(0).numpy(), yt.squeeze(0).numpy()
        if self.channel_dropout > 0 and np.random.rand() < self.channel_dropout and x.shape[0] > 0:
            x[np.random.randint(x.shape[0])] = CHANNEL_DROP_FILL
        if self.elastic > 0:
            xt = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
            yt = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
            xt, yt = elastic_deform(xt, yt, self.elastic)
            x, y = xt.squeeze(0).numpy(), yt.squeeze(0).numpy()
        x = (x - 0.5) / 0.25
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class GpuTiles:
    """Тайлы кэша целиком в памяти видеокарты: батч — индексы, аугментации и распаковка масок — на GPU.

    Поштучная подготовка на CPU (DataLoader в Windows — один поток) оставляла
    видеокарту недогруженной; здесь процессор в цикле обучения не участвует.
    """

    def __init__(self, cache: Path, rows: list[dict], n_classes: int, flips: bool, device, memory_format=None,
                 copy_paste: float = 0.0, paste_channels: tuple[int, ...] = (), class_bits: list[int] | None = None,
                 photo: bool = False, scale: float = 0.0, channel_dropout: float = 0.0,
                 paste_blend: float = 0.0, paste_scale: tuple[float, float] = (1.0, 1.0),
                 elastic: float = 0.0, precompute_target=None,
                 relief: dict | None = None, sample_weights=None):
        import torch

        self.relief = relief or {}
        self.sample_weights = None
        if sample_weights is not None:
            self.sample_weights = torch.as_tensor(np.asarray(sample_weights), dtype=torch.float32, device=device)
        self.memory_format = memory_format or torch.contiguous_format

        by_shard: dict[int, list[int]] = {}
        for r in rows:
            by_shard.setdefault(int(r["shard"]), []).append(int(r["idx"]))
        imgs, msks = [], []
        for s, idx in sorted(by_shard.items()):
            imgs.append(np.asarray(np.load(cache / f"images_{s:04d}.npy", mmap_mode="r")[idx]))
            msks.append(np.asarray(np.load(cache / f"masks_{s:04d}.npy", mmap_mode="r")[idx]))
        self.x = torch.from_numpy(np.concatenate(imgs)).to(device)
        self.m = torch.from_numpy(np.concatenate(msks)).to(device)
        bits = class_bits if class_bits is not None else list(range(n_classes))
        self.shift = torch.tensor(bits, device=device, dtype=self.m.dtype).view(1, -1, 1, 1)
        self.flips, self.device = flips, device
        self.copy_paste, self.paste_channels = copy_paste, paste_channels
        self.photo, self.scale = photo, scale
        self.channel_dropout, self.paste_blend = channel_dropout, paste_blend
        self.paste_scale, self.elastic = paste_scale, elastic
        self.y_pre = None
        relief_zoom = tuple(self.relief.get("zoom", (1.0, 1.0))) != (1.0, 1.0) and self.relief.get("p", 0) > 0
        if precompute_target is not None and scale <= 0 and copy_paste <= 0 and elastic <= 0 and not relief_zoom:
            chunks = []
            for start in range(0, len(self.m), 256):
                ym = ((self.m[start:start + 256].unsqueeze(1) >> self.shift) & 1).float()
                chunks.append(precompute_target(ym))
            self.y_pre = torch.cat(chunks, dim=0) if chunks else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @staticmethod
    def nbytes(cache: Path, rows: list[dict]) -> int:
        """Объём выбранных тайлов и масок в байтах (оценка до загрузки)."""
        img = np.load(cache / "images_0000.npy", mmap_mode="r")
        msk = np.load(cache / "masks_0000.npy", mmap_mode="r")
        return len(rows) * (img[0].nbytes + msk[0].nbytes)

    def get(self, idx, augment: bool):
        import torch

        x = self.x[idx].float().div_(255.0)
        if self.y_pre is not None and (not augment or (self.scale <= 0 and self.copy_paste <= 0 and self.elastic <= 0)):
            y = self.y_pre[idx]
        else:
            y = ((self.m[idx].unsqueeze(1) >> self.shift) & 1).float()
        if augment and self.flips:
            # два отражения и транспонирование с p = 0,5 дают все 8 поворотов и отражений квадрата
            for op in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.transpose(-1, -2)):
                w = (torch.rand(len(idx), device=self.device) < 0.5).view(-1, 1, 1, 1)
                x, y = torch.where(w, op(x), x), torch.where(w, op(y), y)
        if augment and self.relief.get("p", 0) > 0:
            x, y = relief_aug(x, y, self.relief.get("li_tags"), tuple(self.relief.get("height", (1.0, 1.0))),
                              tuple(self.relief.get("zoom", (1.0, 1.0))), float(self.relief["p"]))
        if augment and self.scale > 0:
            x, y = resample_scale(x, y, self.scale)
        if augment and self.copy_paste > 0 and self.paste_channels:
            x, y = paste_objects(x, y, self.paste_channels, self.copy_paste,
                                 blend=self.paste_blend, scale=self.paste_scale)
        if augment and self.channel_dropout > 0:
            x = drop_channel(x, self.channel_dropout)
        if augment and self.elastic > 0:
            x, y = elastic_deform(x, y, self.elastic)
        if augment and self.photo:
            x = photometric(x)
        x = x.sub(0.5).div(0.25)
        return x.contiguous(memory_format=self.memory_format), y

    def batches(self, batch: int, shuffle: bool, drop_last: bool):
        import torch

        if shuffle and self.sample_weights is not None:
            # балансировка участков: выборка с возвращением, длина эпохи прежняя
            order = torch.multinomial(self.sample_weights, len(self), replacement=True)
        else:
            order = torch.randperm(len(self), device=self.device) if shuffle else torch.arange(len(self), device=self.device)
        stop = len(self) - batch + 1 if drop_last else len(self)
        for i in range(0, max(stop, 0), batch):
            yield self.get(order[i:i + batch], augment=shuffle)


def absent_class_weight(target, valid):
    """Вес пикселей по классам. valid — 1, если у образца есть источник этого класса.

    Идея: участок без геофизики не должен учить голову находок, что находок нет нигде.
    valid формы (N, K) или (K,). Там, где 0, вес канала обнуляется. Без valid обучение
    не меняется: функция сама по себе в цикл не включена.
    """

    if valid.ndim == 1:
        valid = valid.view(1, -1)
    if valid.shape[-1] != target.shape[1]:
        raise ValueError(f"классов в valid {valid.shape[-1]}, в разметке {target.shape[1]}")
    view = [1] * target.ndim
    view[0] = valid.shape[0] if valid.shape[0] == target.shape[0] else 1
    view[1] = target.shape[1]
    return valid.to(dtype=target.dtype, device=target.device).view(view)


def dice_loss(logits, target, eps: float = 1.0, weight=None):
    """Мягкий Dice. В среднее входят только классы, у которых в батче есть пиксели.

    Пустой класс при нулевом предсказании даёт Dice 1 и разбавляет промах по
    редкому кургану. На спутнике из-за этого проверка по потере выбирала модель,
    которая не рисует ни одного кургана, хотя по пиксельному F1 она была хуже.
    """
    dims = tuple(d for d in range(target.ndim) if d != 1)
    p = logits.sigmoid()
    if weight is not None:
        p = p * weight
        target = target * weight
    inter = (p * target).sum(dim=dims)
    union = p.sum(dim=dims) + target.sum(dim=dims)
    present = target.sum(dim=dims) > 0
    dice = (2 * inter + eps) / (union + eps)
    if bool(present.any()):
        return 1.0 - dice[present].mean()
    return logits.sum() * 0


def augment_batch(x, y, scale: float = 0.0, photo: float = 0.0):
    """Масштаб и яркость батча на устройстве. Оба нуля — батч не меняется."""
    import torch
    import torch.nn.functional as F

    if scale > 0:
        lo, hi = math.log(1.0 / (1.0 + scale)), math.log(1.0 + scale)
        s = math.exp(float(torch.empty(1).uniform_(lo, hi)))
        t = x.shape[-1]
        n = max(8, round(t * s))
        if n != t:
            x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(n, n), mode="nearest")
            if n > t:
                o = (n - t) // 2
                x, y = x[..., o:o + t, o:o + t], y[..., o:o + t, o:o + t]
            else:
                a = (t - n) // 2
                pad = (a, t - n - a, a, t - n - a)
                x = F.pad(x, pad, mode="replicate")
                y = F.pad(y, pad, value=0.0)
    if photo > 0:
        b = x.shape[0]
        dev = x.device
        x01 = (x * 0.25 + 0.5).clamp(0, 1)
        gamma = torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(-photo, photo))
        contrast = 1 + torch.empty(b, 1, 1, 1, device=dev).uniform_(-photo, photo)
        bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(-photo, photo)
        mean = x01.mean(dim=(-1, -2), keepdim=True)
        x01 = ((x01.pow(gamma) - mean) * contrast + mean + bright).clamp(0, 1)
        x = (x01 - 0.5) / 0.25
    return x.contiguous(), y


def finetune_keeps(f1_mean: float, best_f1: float, val_loss: float, best_loss: float,
                   floor_rec: float | None = None, baseline_floor: float | None = None) -> bool:
    """Можно ли заменить чекпойнт этой эпохой.

    Доучивание сданной сети: если на пороге 0,15 пропала хотя бы половина попаданий
    проверки, эпоху не берём, даже когда суммарная потеря стала меньше.
    Обучение с нуля (baseline_floor is None) — прежнее правило: выше F1,
    при равенстве F1 — ниже потеря.
    """
    if baseline_floor is not None and baseline_floor > 0 and floor_rec is not None and floor_rec < 0.5 * baseline_floor:
        return False
    return f1_mean > best_f1 + 1e-6 or (abs(f1_mean - best_f1) <= 1e-6 and val_loss < best_loss)


def train(cache: str, out: str, encoder: str = "resnet18", epochs: int = 20, batch: int = 16,
          lr: float = 1e-3, val_families: list[str] | None = None, workers: int = 2,
          pretrained: bool = True, limit: int | None = None, amp: bool = True,
          gpu_data: bool | None = None, copy_paste: float = 0.0, seed: int | None = None,
          target: str = "mask", center_sigma_m: float = 4.0, center_radius_m: float = 6.0,
          center_loss_name: str = "bce", all_data: bool = False, min_dist_m: float = 12.0,
          center_spike: bool = True, photo: bool = False, scale: float = 0.0,
          arch: str = "Unet", ema: float = 0.0, scale_aug: float = 0.0, photo_aug: float = 0.0,
          flips_mode: str = "auto", center_edge_ignore: bool = False, use_ignore: bool = False,
          channel_dropout: float = 0.0, paste_blend: float = 0.0,
          paste_scale: tuple[float, float] = (1.0, 1.0), elastic: float = 0.0,
          small_object_weight: float = 1.0, device: str | None = None,
          large_kurgan_weight: float = 1.0, large_damaged_weight: float = 1.0, large_area_m2: float = 1000.0,
          require_gpu: bool = False, min_gpus: int = 0, ema_warmup: bool = False,
          select_min_epoch: int = 1, center_match_m: float = 0.0,
          center_peak_thrs: tuple = (0.5,), center_pre_flip: bool = False,
          within_site_rank: float = 0.0, init_weights: str | None = None,
          relief_height: tuple = (1.0, 1.0), relief_zoom: tuple = (1.0, 1.0), relief_p: float = 0.0,
          site_balance: float = 0.0, center_pos_floor: int = 0) -> dict:
    """Обучает модель. Чекпойнт — лучший F1 на проверке, не минимум потери.

    target="mask" — бинарная маска, F1 по пикселям. target="centers" — гауссиана
    в центре каждой насыпи, F1 по пикам. Второй режим для курганов на снимке:
    маска ставит внутреннюю точку не туда, куда смотрит метрика.
    all_data пишет в файл каждую эпоху, так что на диске остаётся последняя:
    число эпох тогда берётся из уже посчитанных фолдов.
    gpu_data: кэш целиком в видеопамяти (GpuTiles); None — если влезает в четверть свободной.
    """
    if small_object_weight != 1.0 and target != "centers":
        raise ValueError("вес мелких насыпей входит в потерю карты центров: нужен target=centers")
    large_on = large_kurgan_weight != 1.0 or large_damaged_weight != 1.0
    if large_on and target != "centers":
        raise ValueError("вес крупных насыпей (sp_large) входит в потерю карты центров: нужен target=centers")
    if large_on and small_object_weight != 1.0:
        raise ValueError("sp_large: вес крупных и вес мелких насыпей одновременно не поддерживаются")
    if within_site_rank < 0:
        raise ValueError("within_site_rank: коэффициент не может быть отрицательным")
    if within_site_rank > 0 and (target != "centers" or center_loss_name != "focal"):
        raise ValueError("порядок пиков на кадре добавляется только к фокальной потере карты центров")
    if init_weights and not Path(init_weights).is_file():
        raise FileNotFoundError(init_weights)
    no_pre = small_object_weight != 1.0 or large_on
    import segmentation_models_pytorch as smp
    import torch
    from torch.utils.data import DataLoader

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    cache_p = Path(cache)
    meta = json.loads((cache_p / "meta.json").read_text(encoding="utf-8"))
    rows = load_index(cache_p)
    if limit:
        rows = rows[:limit]
    val_families = val_families or []
    tr_rows = [r for r in rows if r["family"] not in val_families]
    va_rows = [r for r in rows if r["family"] in val_families]
    if val_families and not va_rows and not all_data:
        # «--val-families auto» или опечатка раньше молча давали 10 % хвоста индекса (утечка семьи).
        raise ValueError(f"--val-families {val_families}: ни одного тайла этих семей в {cache_p}/index.csv")
    if not va_rows:  # без проверки по семьям — 10 % хвоста (только для отладки)
        cut = max(1, len(rows) // 10)
        tr_rows, va_rows = rows[:-cut], rows[-cut:]
    if all_data:
        # Сдача учится на всех тайлах. Эпоха задана числом эпох (медиана фолдов),
        # а не F1 на куске тех же данных.
        tr_rows = list(rows)
        va_rows = rows[: max(1, min(64, len(rows)))]
    if target not in ("mask", "centers"):
        raise ValueError(f"target={target!r}: ожидается mask или centers")
    center = target == "centers"
    sigma_px = center_sigma_m / float(meta["res"]) if center else 0.0
    k = len(meta["classes"])
    class_bits = meta.get("class_bits")
    ign_bit = meta.get("ignore_bit") if use_ignore else None
    if use_ignore:
        if ign_bit is None:
            raise ValueError("в кэше нет бита «не учить»: кэш собирается с --ignore-other-layers")
        if target != "centers":
            raise ValueError("бит «не учить» поддержан только для target=centers")
        class_bits = list(class_bits if class_bits is not None else range(k)) + [int(ign_bit)]
    c = int(meta.get("channels") or np.load(cache_p / "images_0000.npy", mmap_mode="r").shape[1])
    from .infer_nn import fp16_ok

    must_gpu = bool(require_gpu or os.environ.get("EXPDS_REQUIRE_GPU", "0") == "1"
                    or (device is not None and str(device).startswith("cuda")))
    if must_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            f"Требуется GPU (device={device!r}, require_gpu={require_gpu}), но torch.cuda.is_available() == False! "
            "Отказ от тихого отката на CPU."
        )
    n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if must_gpu and n_dev < max(1, int(min_gpus)):
        raise RuntimeError(f"Требуется не меньше {max(1, int(min_gpus))} GPU, видно {n_dev}. Отказ от тихого отката на CPU.")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if must_gpu and dev.type != "cuda":
        raise RuntimeError(f"--require-gpu, но выбрано устройство {dev}")
    if dev.type == "cuda":
        if must_gpu and dev.index is not None and dev.index >= n_dev:
            raise RuntimeError(f"Запрошен {dev}, а видно {n_dev} GPU: переназначение запрещено при --require-gpu")
        if dev.index is not None and n_dev > 0 and dev.index >= n_dev:
            mapped = dev.index % n_dev
            log.warning("Запрошен %s при %d доступных GPU; переназначаю на cuda:%d", dev, n_dev, mapped)
            dev = torch.device(f"cuda:{mapped}")
        if dev.index is not None:
            torch.cuda.set_device(dev)
    use_amp = amp and fp16_ok(dev)
    if amp and dev.type == "cuda" and not use_amp:
        log.warning("%s: FP16 отключён (на GTX 16xx он даёт NaN), обучение в FP32", torch.cuda.get_device_name(dev))
    # channels_last ускоряет только тензорные ядра в FP16; в FP32 на GTX 1650 он в 4,4 раза медленнее
    fmt = torch.channels_last if use_amp else torch.contiguous_format
    enc_w = None if init_weights or not pretrained else "imagenet"
    if arch == "Unet":
        model = smp.Unet(encoder_name=encoder, encoder_weights=enc_w,
                         in_channels=c, classes=k).to(dev).to(memory_format=fmt)
    else:
        model = smp.create_model(arch, encoder_name=encoder, encoder_weights=enc_w,
                                 in_channels=c, classes=k).to(dev).to(memory_format=fmt)
    ema_model = None
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    if init_weights:
        ck = torch.load(init_weights, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or "state_dict" not in ck:
            raise ValueError(f"{init_weights}: нужен чекпойнт со state_dict")
        got = list(ck.get("classes") or [])
        if got != list(meta["classes"]):
            raise ValueError(f"{init_weights}: классы {got} не совпали с кэшем {list(meta['classes'])}")
        model.load_state_dict(ck["state_dict"])
        log.info("старт с %s, смещение головы не сбрасывается", init_weights)
    elif center and center_loss_name == "focal":
        # смещение головы ставится до копии в EMA, чтобы копия стартовала с тем же prior 0,01.
        # При --init-weights его нет: иначе затирается голова сданной карты.
        from .centers import init_focal_bias
        init_focal_bias(model)
    if must_gpu:
        bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type != "cuda"]
        if bad:
            raise RuntimeError(f"Модель не на GPU: {bad[:3]}")
    if ema > 0:
        if ema_warmup:
            # decay_t = min(ema, (1 + t) / (10 + t)) — при 1,3–1,7 тыс. шагов 0,999 без разгона
            # оставляет 19–28 % веса первых шагов (0,999^1290 = 0,275).
            _ema = float(ema)

            @torch.no_grad()
            def _avg_fn(avg_p, cur_p, num_averaged):
                t = float(num_averaged)
                d = min(_ema, (1.0 + t) / (10.0 + t))
                if not avg_p.is_floating_point():
                    return cur_p
                return avg_p * d + cur_p * (1.0 - d)

            ema_model = AveragedModel(model, avg_fn=_avg_fn, use_buffers=True)
        else:
            ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(ema)), use_buffers=True)
    # auto — как в кэше: повороты и отражения у всего, кроме лидара с направленной отмывкой.
    # none — без них: на снимке тень кургана лежит против солнца, и поворот на 90° учит сеть
    # не смотреть на этот признак (у лидара с отмывкой g повороты выключены по той же причине).
    flips = {"auto": meta.get("flips", False), "none": False, "all": True}[flips_mode]
    torch.backends.cudnn.benchmark = dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if gpu_data is None:
        if dev.type == "cuda":
            free_vram = torch.cuda.mem_get_info(dev)[0]
            vram_frac = 0.55 if free_vram >= 12 * 1024**3 else 0.25
            gpu_data = GpuTiles.nbytes(cache_p, rows) < vram_frac * free_vram
        else:
            gpu_data = False
    ch = paste_channels(meta["classes"]) if copy_paste > 0 else ()
    if copy_paste > 0 and not gpu_data:
        log.warning("copy-paste выключен: тайлы не в видеопамяти")
        ch = ()
    pre_target = None
    if center and not no_pre and scale_aug <= 0:
        from .centers import gaussians_from_binary as _gfb

        def _pre_target(ym):
            src = ym[:, :k]
            t = _gfb(src, sigma_px, by_radius=center_sigma_m <= 0,
                     spike=center_spike, edge_ignore=center_edge_ignore)
            if ign_bit is not None:
                t = torch.where((ym[:, k:] > 0.5) & (t < 0.5), torch.full_like(t, -1.0), t)
            return t

        if not center_spike or (not flips) or center_pre_flip:
            # center_pre_flip: цель считается один раз, отражения применяются к ней на GPU.
            # Шип при отражении может сместиться на 1 пиксель (0,5 м) — меньше радиуса метрики 2 м.
            pre_target = _pre_target
    relief = None
    if relief_p > 0:
        if meta.get("markup") != "li" or not meta.get("li_tags"):
            raise ValueError("relief-aug только для кэша лидара с li_tags")
        relief = {"li_tags": list(meta["li_tags"]), "height": tuple(relief_height),
                  "zoom": tuple(relief_zoom), "p": float(relief_p)}
        log.info("рельеф-аугментация: высота %s, масштаб %s, p %.2f, каналы %s",
                 relief["height"], relief["zoom"], relief_p, relief["li_tags"])
    sw = site_weights(tr_rows, site_balance) if site_balance > 0 else None
    if sw is not None:
        log.info("балансировка участков: alpha %.2f, участков %d", site_balance,
                 len({r.get("region", "") for r in tr_rows}))
    if (relief is not None or sw is not None) and not gpu_data and dev.type == "cpu":
        gpu_data = True  # CPU: те же тензоры GpuTiles в ОЗУ (дымовой тест)
    if (relief is not None or sw is not None) and not gpu_data:
        raise RuntimeError("relief-aug / site-balance реализованы только для тайлов в видеопамяти")
    if gpu_data:
        g_tr = GpuTiles(cache_p, tr_rows, k, flips, dev, fmt, copy_paste=copy_paste, paste_channels=ch,
                        class_bits=class_bits, photo=photo, scale=scale,
                        channel_dropout=channel_dropout, paste_blend=paste_blend,
                        paste_scale=paste_scale, elastic=elastic,
                        precompute_target=pre_target, relief=relief, sample_weights=sw)
        g_va = GpuTiles(cache_p, va_rows, k, False, dev, fmt, class_bits=class_bits,
                        precompute_target=_pre_target if (center and not no_pre) else None)
        n_steps = len(g_tr) // batch

        def train_batches():
            return g_tr.batches(batch, shuffle=True, drop_last=True)

        def val_batches():
            return g_va.batches(batch, shuffle=False, drop_last=False)
    else:
        dl_tr = DataLoader(TileDataset(cache_p, tr_rows, k, flips, class_bits, photo=photo, scale=scale,
                                       channel_dropout=channel_dropout, elastic=elastic),
                           batch_size=batch, shuffle=True,
                           num_workers=workers, drop_last=True, pin_memory=dev.type == "cuda",
                           persistent_workers=workers > 0)
        dl_va = DataLoader(TileDataset(cache_p, va_rows, k, False, class_bits), batch_size=batch,
                           num_workers=workers, persistent_workers=workers > 0)
        n_steps = len(dl_tr)

        def to_dev(it):
            for x, y in it:
                yield x.to(dev, non_blocking=True).to(memory_format=fmt), y.to(dev, non_blocking=True)

        def train_batches():
            return to_dev(dl_tr)

        def val_batches():
            return to_dev(dl_va)
    log.info("данные: %s, тайлов обучения %d, проверки %d%s",
             "в видеопамяти" if gpu_data else "DataLoader", len(tr_rows), len(va_rows),
             f", copy-paste {copy_paste:g}" if ch else "")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * max(1, n_steps))
    if init_weights:
        # Короткое доучивание сданной сети: постоянный шаг. Разгон OneCycle рассчитан на старт с нуля
        # и на 1650 за четыре эпохи стёр пики выше рабочего порога.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _step: 1.0)
        log.info("доучивание: постоянный шаг %.3g, без разгона OneCycle", lr)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = torch.nn.BCEWithLogitsLoss()
    if center:
        if center:
            from .centers import (
                center_loss,
                focal_center_loss,
                gaussians_from_binary,
                heatmap_counts,
                small_component_weights,
                within_site_rank_hinge,
            )
            if within_site_rank > 0:
                log.info("порядок пиков на кадре: коэффициент %.3f, не больше 8 ложных, запас 1",
                         within_site_rank)

            def prepare(y):
                src = y[:, :k]
                t = gaussians_from_binary(src, sigma_px, by_radius=center_sigma_m <= 0,
                                          spike=center_spike, edge_ignore=center_edge_ignore)
                if ign_bit is not None:
                    t = torch.where((y[:, k:] > 0.5) & (t < 0.5), torch.full_like(t, -1.0), t)
                return t

            if large_on:
                from .centers import large_component_weights
                _res = float(meta.get("res") or 1.0)
                _shr = float(meta.get("shrink") or 0.0)
                _min_px = float(large_area_m2) * (1.0 - _shr) ** 2 / (_res * _res)
                _cls = list(meta["classes"])[:k]
                _cw = [large_damaged_weight if c == "kurgany_povrezhdennye" else
                       large_kurgan_weight if c.startswith("kurgany") else 1.0 for c in _cls]
                log.info("sp_large: вес крупных насыпей %s, порог %.0f м² эталона = %.1f пикс маски (шаг %.2f, сжатие %.2f)",
                         dict(zip(_cls, _cw)), large_area_m2, _min_px, _res, _shr)

            def batch_loss(logits, y, precomputed: bool = False):
                if precomputed:
                    wmap = None
                elif large_on:
                    wmap = large_component_weights(y[:, :k], _cw, _min_px)
                else:
                    wmap = small_component_weights(y[:, :k], small_object_weight)
                if not precomputed:
                    y = prepare(y)
                loss = (focal_center_loss(logits, y, weight=wmap, pos_floor=center_pos_floor)
                        if center_loss_name == "focal" else center_loss(logits, y, weight=wmap))
                if within_site_rank > 0:
                    loss = loss + within_site_rank * within_site_rank_hinge(logits, y)
                return loss, y
    else:
        def batch_loss(logits, y, precomputed: bool = False):
            return bce(logits, y) + dice_loss(logits.float(), y), y
    best_f1, best_loss, history = -1.0, math.inf, []
    thrs = tuple(float(t) for t in (center_peak_thrs or (0.5,)))
    best_thr = None
    kept_init = False
    saved = False
    baseline_floor = None
    FLOOR_THR = 0.15

    def _floor_recall(net) -> float | None:
        if not (center and init_weights):
            return None
        from .centers import heatmap_counts as _hc
        net.eval()
        match_r = max(2, round(sigma_px)) if center_sigma_m <= 0 else max(1, round(sigma_px))
        if center_match_m > 0:
            match_r = max(1, int(round(center_match_m / float(meta["res"]))))
        tp = fn = 0.0
        with torch.inference_mode():
            for x, y in val_batches():
                va_pre = bool(gpu_data and g_va.y_pre is not None)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    logits = net(x)
                _, y = batch_loss(logits, y, precomputed=va_pre)
                ctp, _, cfn = _hc(logits.float().sigmoid(), y, match_r, thr=FLOOR_THR)
                tp += float(ctp.sum())
                fn += float(cfn.sum())
        return tp / max(tp + fn, 1e-6)

    init_cpu = None
    if init_weights and not all_data:
        init_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if init_weights and center and not all_data:
        baseline_floor = _floor_recall(model)
        log.info("доучивание: полнота пиков на проверке при пороге %.2f до шага — %.3f", FLOOR_THR, baseline_floor)
    checked_dev = not must_gpu
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tl = torch.zeros((), device=dev)
        n = 0
        for x, y in train_batches():
            if not checked_dev:
                if x.device.type != "cuda" or y.device.type != "cuda":
                    raise RuntimeError(f"Батч не на GPU: x={x.device}, y={y.device}")
                checked_dev = True
            if scale_aug > 0 or photo_aug > 0:
                x, y = augment_batch(x, y, scale_aug, photo_aug)
            tr_pre = bool(gpu_data and g_tr.y_pre is not None and scale_aug <= 0)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss, _ = batch_loss(logits, y, precomputed=tr_pre)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            tl += loss.detach() * len(x)
            n += len(x)
        net = ema_model.module if ema_model is not None else model
        net.eval()
        vl = torch.zeros((), device=dev)
        vn = 0
        tp, fp, fn = (torch.zeros(k, device=dev) for _ in range(3))
        ctab = {t: [torch.zeros(k, device=dev) for _ in range(3)] for t in thrs} if center else {}
        floor_tp = torch.zeros(k, device=dev) if (center and init_weights) else None
        floor_fn = torch.zeros(k, device=dev) if floor_tp is not None else None
        with torch.inference_mode():
            for x, y in val_batches():
                va_pre = bool(gpu_data and g_va.y_pre is not None)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    logits = net(x)
                loss, y = batch_loss(logits, y, precomputed=va_pre)
                vl += loss * len(x)
                vn += len(x)
                if center:
                    match_r = max(2, round(sigma_px)) if center_sigma_m <= 0 else max(1, round(sigma_px))
                    if center_match_m > 0:
                        match_r = max(1, int(round(center_match_m / float(meta["res"]))))
                    pr_sig = logits.float().sigmoid()
                    for t_ in thrs:
                        ctp, cfp, cfn = heatmap_counts(pr_sig, y, match_r, thr=t_)
                        ctab[t_][0] += ctp
                        ctab[t_][1] += cfp
                        ctab[t_][2] += cfn
                    if floor_tp is not None:
                        ftp, _, ffn = heatmap_counts(pr_sig, y, match_r, thr=FLOOR_THR)
                        floor_tp += ftp
                        floor_fn += ffn
                else:
                    p = (logits.float().sigmoid() > 0.5).float()
                    tp += (p * y).sum(dim=(0, 2, 3))
                    fp += (p * (1 - y)).sum(dim=(0, 2, 3))
                    fn += ((1 - p) * y).sum(dim=(0, 2, 3))
        vl = float(vl) / max(vn, 1)
        tp, fp, fn = tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy()
        f1_by_thr = {}
        ep_thr = None
        if center:
            # F1 пиков на нескольких порогах; для отбора эпохи — лучший порог (порог с семьи проверки,
            # не с тестовой семьи фолда, т. е. leave-family-out сохраняется).
            best_mean = -2.0
            for t_ in thrs:
                a, b, c_ = (v.cpu().numpy() for v in ctab[t_])
                pr_ = a / np.maximum(a + 0.5 * b, 1e-6)
                rc_ = a / np.maximum(a + c_, 1e-6)
                f_ = 2 * pr_ * rc_ / np.maximum(pr_ + rc_, 1e-6)
                sup_ = (a + c_) > 0
                m_ = float(f_[sup_].mean()) if sup_.any() else -1.0
                f1_by_thr[t_] = round(m_, 4)
                if m_ > best_mean:
                    best_mean, ep_thr, tp, fp, fn = m_, t_, a, b, c_
            prec = tp / np.maximum(tp + 0.5 * fp, 1e-6)
            rec_c = tp / np.maximum(tp + fn, 1e-6)
            f1 = 2 * prec * rec_c / np.maximum(prec + rec_c, 1e-6)
            f1_key = "center_f1"
        else:
            f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
            f1_key = "pixel_f1"
        rec = {"epoch": ep + 1, "train_loss": round(float(tl) / max(n, 1), 4), "val_loss": round(vl, 4),
               f1_key: {cls: round(float(v), 3) for cls, v in zip(meta["classes"], f1)},
               "seconds": round(time.perf_counter() - t0, 1)}
        if center:
            rec["center_f1_mean_by_thr"] = {str(t_): v for t_, v in f1_by_thr.items()}
            rec["peak_thr"] = ep_thr
        history.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
        support = (tp + fn) > 0
        f1_mean = float(f1[support].mean()) if support.any() else -1.0
        floor_rec = None
        if floor_tp is not None:
            floor_rec = float(floor_tp.sum() / torch.clamp(floor_tp.sum() + floor_fn.sum(), min=1e-6))
            rec["floor_recall_0.15"] = round(floor_rec, 4)
        # Чекпойнт — по F1 классов, которые есть в проверке (пиксели или пики центров).
        # Потеря почти целиком про фон и выбирала пустую карту. При доучивании пустая карта
        # ещё и снижает шарнир, поэтому одной потери мало: пики на 0,15 должны остаться.
        better = finetune_keeps(f1_mean, best_f1, vl, best_loss, floor_rec, baseline_floor)
        if ep + 1 < int(select_min_epoch):
            better = False  # эпоха 1–2 с F1 0,002 больше не «лучшая» навсегда
        if all_data:
            better = True
        if better:
            saved = True
            best_f1, best_loss, best_thr = f1_mean, vl, ep_thr
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "encoder": encoder, "arch": arch, "in_channels": c,
                        "classes": meta["classes"], "markup": meta["markup"], "res": meta["res"],
                        "tile": meta["tile"], "li_tags": meta.get("li_tags"),
                        "li_source": meta.get("li_source"), "dem_res": meta.get("dem_res"),
                        "dem_fill": meta.get("dem_fill"), "max_upsample": meta.get("max_upsample"),
                        "ema": float(ema),
                        "target": target, "center_sigma_m": center_sigma_m if center else None,
                        "center_radius_m": center_radius_m if center else None,
                        "min_dist_m": min_dist_m if center else None,
                        "center_edge_ignore": bool(center_edge_ignore) if center else None,
                        "ignore_other_layers": ign_bit is not None,
                        "pixel_units": meta.get("pixel_units"),
                        "photo": photo, "scale": scale,
                        "epoch": ep + 1, "val_peak_thr": ep_thr, "seed": seed,
                        "ema_warmup": bool(ema_warmup), "device": str(dev),
                        "relief_aug": relief, "site_balance": float(site_balance),
                        "val_loss": vl, "kept_init": False}, out)
    if init_cpu is not None and not saved:
        kept_init = True
        log.info("доучивание не обошло стартовые веса по пикам: в файл записан старт")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": init_cpu, "encoder": encoder, "arch": arch, "in_channels": c,
                    "classes": meta["classes"], "markup": meta["markup"], "res": meta["res"],
                    "tile": meta["tile"], "kept_init": True,
                    "target": target, "center_sigma_m": center_sigma_m if center else None}, out)
    return {"best_val_loss": best_loss, "best_pixel_f1": best_f1, "best_peak_thr": best_thr,
            "history": history, "out": out, "kept_init": kept_init,
            "train_tiles": len(tr_rows), "val_tiles": len(va_rows), "device": str(dev)}


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Обучение U-Net на кэше тайлов")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-families", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=2, help="на Windows при ошибках поставьте 0")
    ap.add_argument("--no-pretrained", action="store_true", help="без загрузки весов ImageNet")
    ap.add_argument("--limit", type=int, default=None, help="взять первые N тайлов (отладка)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cpu-data", action="store_true", help="тайлы через DataLoader, а не из видеопамяти")
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="вероятность вставить курган с другого тайла (Simple Copy-Paste)")
    ap.add_argument("--photo", action="store_true",
                    help="на обучении яркость ±0,08 и контраст ±0,15")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="случайный масштаб тайла, доля; вместе с --photo не включать")
    ap.add_argument("--seed", type=int, default=None, help="зерно инициализации и порядка батчей (ансамбль)")
    ap.add_argument("--target", choices=("mask", "centers"), default="mask",
                    help="mask — маска класса; centers — гауссиана в центре объекта")
    ap.add_argument("--center-sigma", type=float, default=4.0, help="ширина гауссианы центра, м")
    ap.add_argument("--center-radius", type=float, default=6.0, help="радиус круга на выходе, м")
    ap.add_argument("--center-min-dist", type=float, default=12.0,
                    help="минимальное расстояние между центрами на выходе, м")
    ap.add_argument("--center-loss", choices=("bce", "focal"), default="bce",
                    help="focal — потеря карты центров Law & Deng / CenterNet")
    ap.add_argument("--all-data", action="store_true",
                    help="все тайлы в обучении, в файл весов пишется последняя эпоха")
    ap.add_argument("--arch", default="Unet",
                    help="декодер smp: Unet, UnetPlusPlus, FPN, DeepLabV3Plus, UPerNet, MAnet")
    ap.add_argument("--ema", type=float, default=0.0, help="экспоненциальное среднее весов, 0 — выключено")
    ap.add_argument("--scale-aug", type=float, default=0.0, help="случайный масштаб батча, 0 — выключено")
    ap.add_argument("--photo-aug", type=float, default=0.0, help="яркость и гамма батча, 0 — выключено")
    ap.add_argument("--flips", choices=("auto", "none", "all"), default="auto",
                    help="повороты и отражения тайлов: auto — как в кэше; none — сохранить направление "
                         "солнца на снимке (опыт 25.10)")
    ap.add_argument("--center-edge-ignore", action="store_true",
                    help="насыпь, обрезанную краем тайла, не учить: без пика, область −1 (отдельный опыт)")
    ap.add_argument("--use-ignore", action="store_true",
                    help="читать бит «не учить» кэша (--ignore-other-layers): объекты других слоёв — не фон")
    ap.add_argument("--channel-dropout", type=float, default=0.0,
                    help="вероятность занулить один канал кадра; 0 — как раньше")
    ap.add_argument("--paste-blend", type=float, default=0.0,
                    help="размер окна размытия края copy-paste; 0 — жёсткая вставка")
    ap.add_argument("--elastic", type=float, default=0.0,
                    help="сила упругого сдвига сетки; 0 — без него")
    ap.add_argument("--device", default=None, help="устройство PyTorch (cuda:0, cuda:1, cpu)")
    ap.add_argument("--require-gpu", action="store_true",
                    help="завершить работу с ошибкой, если GPU недоступен (защита от тихого отката на CPU в облаке)")
    ap.add_argument("--small-object-weight", type=float, default=1.0,
                    help="вес потери насыпи мельче 0,2 %% кадра (карта центров); 1 — без него")
    ap.add_argument("--large-kurgan-weight", type=float, default=1.0,
                    help="sp_large: вес пика кургана от --large-area-m2 (целые и прочие курганы); 1 — выключено")
    ap.add_argument("--large-damaged-weight", type=float, default=1.0,
                    help="sp_large: вес пика повреждённого кургана от --large-area-m2; 1 — выключено")
    ap.add_argument("--large-area-m2", type=float, default=1000.0, help="sp_large: порог площади эталона, м²")
    ap.add_argument("--min-gpus", type=int, default=0, help="минимум видимых GPU при --require-gpu")
    ap.add_argument("--ema-warmup", action="store_true", help="EMA с разгоном decay=min(ema,(1+t)/(10+t))")
    ap.add_argument("--select-min-epoch", type=int, default=1, help="чекпойнт не раньше этой эпохи")
    ap.add_argument("--center-match-m", type=float, default=0.0,
                    help="радиус совпадения пиков на проверке, м (0 — как раньше: 2 пикселя)")
    ap.add_argument("--center-peak-thrs", type=float, nargs="*", default=[0.5],
                    help="пороги пика для F1 на проверке; отбор эпохи по лучшему")
    ap.add_argument("--center-pre-flip", action="store_true",
                    help="цель центров считать один раз и отражать на GPU (ускорение эпохи)")
    ap.add_argument("--within-site-rank", type=float, default=0.0,
                    help="коэффициент шарнира «центр эталона выше ложных пиков кадра»; 0 — выключено, пилот — 1")
    ap.add_argument("--init-weights", default=None,
                    help="чекпойнт для доучивания; без него голова фокальной карты сбрасывается на prior 0,01")
    ap.add_argument("--relief-height", type=float, nargs=2, default=(1.0, 1.0), metavar=("LO", "HI"),
                    help="лидар: множитель высоты рельефа (лог-равномерно), напр. 0.5 2.2; 1 1 — выкл.")
    ap.add_argument("--relief-zoom", type=float, nargs=2, default=(1.0, 1.0), metavar=("LO", "HI"),
                    help="лидар: масштаб кадра с пересчётом уклона, напр. 0.7 1.45; 1 1 — выкл.")
    ap.add_argument("--relief-p", type=float, default=0.0, help="доля кадров с рельеф-аугментацией; 0 — выкл.")
    ap.add_argument("--site-balance", type=float, default=0.0,
                    help="вес тайла ∝ n_участка^(−alpha); 0 — выкл., 0.5 — квадратный корень")
    ap.add_argument("--center-pos-floor", type=int, default=0,
                    help="пол числа пиков в знаменателе фокальной потери; 0 — как раньше")
    ap.add_argument("--dry-run", action="store_true", help="только разобрать аргументы и выйти")
    args = ap.parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False))
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = train(args.cache, args.out, args.encoder, args.epochs, args.batch, args.lr, args.val_families,
                args.workers, not args.no_pretrained, args.limit, not args.no_amp,
                gpu_data=False if args.cpu_data else None, copy_paste=args.copy_paste, seed=args.seed,
                target=args.target, center_sigma_m=args.center_sigma, center_radius_m=args.center_radius,
                center_loss_name=args.center_loss, all_data=args.all_data, min_dist_m=args.center_min_dist,
                photo=args.photo, scale=args.scale, arch=args.arch, ema=args.ema,
                scale_aug=args.scale_aug, photo_aug=args.photo_aug, flips_mode=args.flips,
                center_edge_ignore=args.center_edge_ignore, use_ignore=args.use_ignore,
                channel_dropout=args.channel_dropout, paste_blend=args.paste_blend,
                elastic=args.elastic, small_object_weight=args.small_object_weight, device=args.device, require_gpu=args.require_gpu,
                large_kurgan_weight=args.large_kurgan_weight, large_damaged_weight=args.large_damaged_weight,
                large_area_m2=args.large_area_m2,
                min_gpus=args.min_gpus, ema_warmup=args.ema_warmup, select_min_epoch=args.select_min_epoch,
                center_match_m=args.center_match_m, center_peak_thrs=tuple(args.center_peak_thrs),
                center_pre_flip=args.center_pre_flip, within_site_rank=args.within_site_rank,
                init_weights=args.init_weights,
                relief_height=tuple(args.relief_height), relief_zoom=tuple(args.relief_zoom),
                relief_p=args.relief_p, site_balance=args.site_balance,
                center_pos_floor=args.center_pos_floor)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
