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
                 elastic: float = 0.0, precompute_target=None):
        import torch

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
        if precompute_target is not None and scale <= 0 and copy_paste <= 0 and elastic <= 0:
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

        order = torch.randperm(len(self), device=self.device) if shuffle else torch.arange(len(self), device=self.device)
        stop = len(self) - batch + 1 if drop_last else len(self)
        for i in range(0, max(stop, 0), batch):
            yield self.get(order[i:i + batch], augment=shuffle)


def dice_loss(logits, target, eps: float = 1.0):
    """Мягкий Dice. В среднее входят только классы, у которых в батче есть пиксели.

    Пустой класс при нулевом предсказании даёт Dice 1 и разбавляет промах по
    редкому кургану. На спутнике из-за этого проверка по потере выбирала модель,
    которая не рисует ни одного кургана, хотя по пиксельному F1 она была хуже.
    """
    dims = tuple(d for d in range(target.ndim) if d != 1)
    p = logits.sigmoid()
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
          small_object_weight: float = 1.0, device: str | None = None) -> dict:
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

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev)
    use_amp = amp and fp16_ok(dev)
    if amp and dev.type == "cuda" and not use_amp:
        log.warning("%s: FP16 отключён (на GTX 16xx он даёт NaN), обучение в FP32", torch.cuda.get_device_name(dev))
    # channels_last ускоряет только тензорные ядра в FP16; в FP32 на GTX 1650 он в 4,4 раза медленнее
    fmt = torch.channels_last if use_amp else torch.contiguous_format
    if arch == "Unet":
        model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                         in_channels=c, classes=k).to(dev).to(memory_format=fmt)
    else:
        model = smp.create_model(arch, encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                                 in_channels=c, classes=k).to(dev).to(memory_format=fmt)
    ema_model = None
    if ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(float(ema)), use_buffers=True)
    if center and center_loss_name == "focal":
        from .centers import init_focal_bias
        init_focal_bias(model)
    # auto — как в кэше: повороты и отражения у всего, кроме лидара с направленной отмывкой.
    # none — без них: на снимке тень кургана лежит против солнца, и поворот на 90° учит сеть
    # не смотреть на этот признак (у лидара с отмывкой g повороты выключены по той же причине).
    flips = {"auto": meta.get("flips", False), "none": False, "all": True}[flips_mode]
    torch.backends.cudnn.benchmark = dev.type == "cuda"
    if gpu_data is None:
        gpu_data = dev.type == "cuda" and GpuTiles.nbytes(cache_p, rows) < 0.25 * torch.cuda.mem_get_info(dev)[0]
    ch = paste_channels(meta["classes"]) if copy_paste > 0 else ()
    if copy_paste > 0 and not gpu_data:
        log.warning("copy-paste выключен: тайлы не в видеопамяти")
        ch = ()
    pre_target = None
    if center and small_object_weight == 1.0 and scale_aug <= 0:
        from .centers import gaussians_from_binary as _gfb

        def _pre_target(ym):
            src = ym[:, :k]
            t = _gfb(src, sigma_px, by_radius=center_sigma_m <= 0,
                     spike=center_spike, edge_ignore=center_edge_ignore)
            if ign_bit is not None:
                t = torch.where((ym[:, k:] > 0.5) & (t < 0.5), torch.full_like(t, -1.0), t)
            return t

        if not center_spike or (not flips):
            pre_target = _pre_target
    if gpu_data:
        g_tr = GpuTiles(cache_p, tr_rows, k, flips, dev, fmt, copy_paste=copy_paste, paste_channels=ch,
                        class_bits=class_bits, photo=photo, scale=scale,
                        channel_dropout=channel_dropout, paste_blend=paste_blend,
                        paste_scale=paste_scale, elastic=elastic,
                        precompute_target=pre_target)
        g_va = GpuTiles(cache_p, va_rows, k, False, dev, fmt, class_bits=class_bits,
                        precompute_target=_pre_target if (center and small_object_weight == 1.0) else None)
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
            )

            def prepare(y):
                src = y[:, :k]
                t = gaussians_from_binary(src, sigma_px, by_radius=center_sigma_m <= 0,
                                          spike=center_spike, edge_ignore=center_edge_ignore)
                if ign_bit is not None:
                    t = torch.where((y[:, k:] > 0.5) & (t < 0.5), torch.full_like(t, -1.0), t)
                return t

            def batch_loss(logits, y, precomputed: bool = False):
                wmap = None if precomputed else small_component_weights(y[:, :k], small_object_weight)
                if not precomputed:
                    y = prepare(y)
                loss = (focal_center_loss(logits, y, weight=wmap) if center_loss_name == "focal"
                        else center_loss(logits, y, weight=wmap))
                return loss, y
    else:
        def batch_loss(logits, y, precomputed: bool = False):
            return bce(logits, y) + dice_loss(logits.float(), y), y
    best_f1, best_loss, history = -1.0, math.inf, []
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tl = torch.zeros((), device=dev)
        n = 0
        for x, y in train_batches():
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
                    ctp, cfp, cfn = heatmap_counts(logits.float().sigmoid(), y, match_r)
                    tp += ctp
                    fp += cfp
                    fn += cfn
                else:
                    p = (logits.float().sigmoid() > 0.5).float()
                    tp += (p * y).sum(dim=(0, 2, 3))
                    fp += (p * (1 - y)).sum(dim=(0, 2, 3))
                    fn += ((1 - p) * y).sum(dim=(0, 2, 3))
        vl = float(vl) / max(vn, 1)
        tp, fp, fn = tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy()
        if center:
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
        history.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
        support = (tp + fn) > 0
        f1_mean = float(f1[support].mean()) if support.any() else -1.0
        # Чекпойнт — по F1 классов, которые есть в проверке (пиксели или пики центров).
        # Потеря BCE почти целиком про фон и выбирала пустую карту.
        better = f1_mean > best_f1 + 1e-6 or (abs(f1_mean - best_f1) <= 1e-6 and vl < best_loss)
        if all_data:
            better = True
        if better:
            best_f1, best_loss = f1_mean, vl
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
                        "epoch": ep + 1,
                        "val_loss": vl}, out)
    return {"best_val_loss": best_loss, "best_pixel_f1": best_f1, "history": history, "out": out,
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
    ap.add_argument("--small-object-weight", type=float, default=1.0,
                    help="вес потери насыпи мельче 0,2 %% кадра (карта центров); 1 — без него")
    args = ap.parse_args()
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
                elastic=args.elastic, small_object_weight=args.small_object_weight, device=args.device)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
