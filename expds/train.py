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


def load_index(cache: Path) -> list[dict]:
    """Строки index.csv кэша."""
    with (cache / "index.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paste_channels(classes: list[str]) -> tuple[int, ...]:
    """Индексы классов, которые copy-paste переносит (курганы)."""
    return tuple(i for i, name in enumerate(classes) if name in PASTE_CLASSES)


def paste_objects(x, y, channels: tuple[int, ...], p: float):
    """Simple Copy-Paste: пиксели объекта с тайла-донора заменяют фон приёмника.

    x — N×C×H×W, y — N×K×H×W. Объект — пиксели каналов `channels`. Вставка только
    на тайл без кургана (с вероятностью p): плотная группа эталона не затирается,
    а пустой фон, где модель даёт ложные насыпи, получает настоящую форму.
    Донор — случайный тайл батча, где объект есть. Насыпь сдвигается так, чтобы её
    центр масс попал в случайный пиксель. Фон донора не копируется. Ghiasi et al., CVPR 2021.
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
    return torch.where(alpha, sx, x), torch.where(alpha, sy, y)


class TileDataset:
    """Тайлы из .npy-шардов (memmap); маски — биты классов → K каналов float."""

    def __init__(self, cache: Path, rows: list[dict], n_classes: int, flips: bool):
        self.cache, self.rows, self.k, self.flips = cache, rows, n_classes, flips
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
        y = np.stack([(bits >> k) & 1 for k in range(self.k)]).astype(np.float32)
        if self.flips:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
        x = (x - 0.5) / 0.25
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


class GpuTiles:
    """Тайлы кэша целиком в памяти видеокарты: батч — индексы, аугментации и распаковка масок — на GPU.

    Поштучная подготовка на CPU (DataLoader в Windows — один поток) оставляла
    видеокарту недогруженной; здесь процессор в цикле обучения не участвует.
    """

    def __init__(self, cache: Path, rows: list[dict], n_classes: int, flips: bool, device, memory_format=None,
                 copy_paste: float = 0.0, paste_channels: tuple[int, ...] = ()):
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
        self.shift = torch.arange(n_classes, device=device, dtype=self.m.dtype).view(1, -1, 1, 1)
        self.flips, self.device = flips, device
        self.copy_paste, self.paste_channels = copy_paste, paste_channels

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

        x = self.x[idx].float().div_(255.0).sub_(0.5).div_(0.25)
        y = ((self.m[idx].unsqueeze(1) >> self.shift) & 1).float()
        if augment and self.flips:
            # два отражения и транспонирование с p = 0,5 дают все 8 поворотов и отражений квадрата
            for op in (lambda t: t.flip(-1), lambda t: t.flip(-2), lambda t: t.transpose(-1, -2)):
                w = (torch.rand(len(idx), device=self.device) < 0.5).view(-1, 1, 1, 1)
                x, y = torch.where(w, op(x), x), torch.where(w, op(y), y)
        if augment and self.copy_paste > 0 and self.paste_channels:
            x, y = paste_objects(x, y, self.paste_channels, self.copy_paste)
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


def train(cache: str, out: str, encoder: str = "resnet18", epochs: int = 20, batch: int = 16,
          lr: float = 1e-3, val_families: list[str] | None = None, workers: int = 2,
          pretrained: bool = True, limit: int | None = None, amp: bool = True,
          gpu_data: bool | None = None, copy_paste: float = 0.0, seed: int | None = None,
          target: str = "mask", center_sigma_m: float = 4.0, center_radius_m: float = 6.0,
          center_loss_name: str = "bce") -> dict:
    """Обучает модель. Чекпойнт — лучший F1 на проверке, не минимум потери.

    target="mask" — бинарная маска, F1 по пикселям. target="centers" — гауссиана
    в центре каждой насыпи, F1 по пикам. Второй режим для курганов на снимке:
    маска ставит внутреннюю точку не туда, куда смотрит метрика.
    gpu_data: кэш целиком в видеопамяти (GpuTiles); None — если влезает в четверть свободной.
    """
    import torch
    import segmentation_models_pytorch as smp
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
    if target not in ("mask", "centers"):
        raise ValueError(f"target={target!r}: ожидается mask или centers")
    center = target == "centers"
    sigma_px = center_sigma_m / float(meta["res"]) if center else 0.0
    k = len(meta["classes"])
    c = int(meta.get("channels") or np.load(cache_p / "images_0000.npy", mmap_mode="r").shape[1])
    from .infer_nn import fp16_ok

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and fp16_ok(dev)
    if amp and dev.type == "cuda" and not use_amp:
        log.warning("%s: FP16 отключён (на GTX 16xx он даёт NaN), обучение в FP32", torch.cuda.get_device_name(dev))
    # channels_last ускоряет только тензорные ядра в FP16; в FP32 на GTX 1650 он в 4,4 раза медленнее
    fmt = torch.channels_last if use_amp else torch.contiguous_format
    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                     in_channels=c, classes=k).to(dev).to(memory_format=fmt)
    flips = meta.get("flips", False)
    torch.backends.cudnn.benchmark = dev.type == "cuda"
    if gpu_data is None:
        gpu_data = dev.type == "cuda" and GpuTiles.nbytes(cache_p, rows) < 0.25 * torch.cuda.mem_get_info(dev)[0]
    ch = paste_channels(meta["classes"]) if copy_paste > 0 else ()
    if copy_paste > 0 and not gpu_data:
        log.warning("copy-paste выключен: тайлы не в видеопамяти")
        ch = ()
    if gpu_data:
        g_tr = GpuTiles(cache_p, tr_rows, k, flips, dev, fmt, copy_paste=copy_paste, paste_channels=ch)
        g_va = GpuTiles(cache_p, va_rows, k, False, dev, fmt)
        n_steps = len(g_tr) // batch

        def train_batches():
            return g_tr.batches(batch, shuffle=True, drop_last=True)

        def val_batches():
            return g_va.batches(batch, shuffle=False, drop_last=False)
    else:
        dl_tr = DataLoader(TileDataset(cache_p, tr_rows, k, flips), batch_size=batch, shuffle=True,
                           num_workers=workers, drop_last=True, pin_memory=dev.type == "cuda")
        dl_va = DataLoader(TileDataset(cache_p, va_rows, k, False), batch_size=batch, num_workers=workers)
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
            from .centers import center_loss, focal_center_loss, gaussians_from_binary, heatmap_counts

            def prepare(y):
                return gaussians_from_binary(y, sigma_px, by_radius=center_sigma_m <= 0)

            def batch_loss(logits, y):
                y = prepare(y)
                loss = focal_center_loss(logits, y) if center_loss_name == "focal" else center_loss(logits, y)
                return loss, y
    else:
        def batch_loss(logits, y):
            return bce(logits, y) + dice_loss(logits.float(), y), y
    best_f1, best_loss, history = -1.0, math.inf, []
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tl = torch.zeros((), device=dev)
        n = 0
        for x, y in train_batches():
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss, _ = batch_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.detach() * len(x)
            n += len(x)
        model.eval()
        vl = torch.zeros((), device=dev)
        vn = 0
        tp, fp, fn = (torch.zeros(k, device=dev) for _ in range(3))
        with torch.inference_mode():
            for x, y in val_batches():
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                loss, y = batch_loss(logits, y)
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
        if better:
            best_f1, best_loss = f1_mean, vl
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "encoder": encoder, "in_channels": c,
                        "classes": meta["classes"], "markup": meta["markup"], "res": meta["res"],
                        "tile": meta["tile"], "li_tags": meta.get("li_tags"),
                        "li_source": meta.get("li_source"), "dem_res": meta.get("dem_res"),
                        "dem_fill": meta.get("dem_fill"), "max_upsample": meta.get("max_upsample"),
                        "target": target, "center_sigma_m": center_sigma_m if center else None,
                        "center_radius_m": center_radius_m if center else None,
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
    ap.add_argument("--seed", type=int, default=None, help="зерно инициализации и порядка батчей (ансамбль)")
    ap.add_argument("--target", choices=("mask", "centers"), default="mask",
                    help="mask — маска класса; centers — гауссиана в центре объекта")
    ap.add_argument("--center-sigma", type=float, default=4.0, help="ширина гауссианы центра, м")
    ap.add_argument("--center-radius", type=float, default=6.0, help="радиус круга на выходе, м")
    ap.add_argument("--center-loss", choices=("bce", "focal"), default="bce",
                    help="focal — потеря карты центров Law & Deng / CenterNet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = train(args.cache, args.out, args.encoder, args.epochs, args.batch, args.lr, args.val_families,
                args.workers, not args.no_pretrained, args.limit, not args.no_amp,
                gpu_data=False if args.cpu_data else None, copy_paste=args.copy_paste, seed=args.seed,
                target=args.target, center_sigma_m=args.center_sigma, center_radius_m=args.center_radius,
                center_loss_name=args.center_loss)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
