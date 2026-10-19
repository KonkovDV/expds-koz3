"""Обучение U-Net на кэше тайлов — режим для видеокарты 4 ГБ (GTX 1650).

Настройки по умолчанию: ResNet18 с весами ImageNet, тайл 256, батч 16,
fp16 (AMP), без тяжёлых аугментаций. Для отмывок лидара повороты и
отражения выключены: они меняют направление «освещения», и модель учит
артефакт. Проверка — по «семьям» участков (--val-families), а не случайно.

    python -m expds.train --cache work/cache/li --out models/li_r18.pt --val-families Гнёздово
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


def load_index(cache: Path) -> list[dict]:
    """Строки index.csv кэша."""
    with (cache / "index.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def dice_loss(logits, target, eps: float = 1.0):
    """Мягкий Dice по классам (среднее по классам, где есть пиксели или предсказания)."""
    p = logits.sigmoid()
    inter = (p * target).sum(dim=(0, 2, 3))
    union = p.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def train(cache: str, out: str, encoder: str = "resnet18", epochs: int = 20, batch: int = 16,
          lr: float = 1e-3, val_families: list[str] | None = None, workers: int = 2,
          pretrained: bool = True, limit: int | None = None, amp: bool = True) -> dict:
    """Обучает модель и сохраняет лучший по валидационной потере чекпойнт."""
    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader

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
    k = len(meta["classes"])
    c = int(meta.get("channels") or np.load(cache_p / "images_0000.npy", mmap_mode="r").shape[1])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and dev.type == "cuda"
    model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                     in_channels=c, classes=k).to(dev).to(memory_format=torch.channels_last)
    dl_tr = DataLoader(TileDataset(cache_p, tr_rows, k, meta.get("flips", False)), batch_size=batch,
                       shuffle=True, num_workers=workers, drop_last=True, pin_memory=dev.type == "cuda")
    dl_va = DataLoader(TileDataset(cache_p, va_rows, k, False), batch_size=batch, num_workers=workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * max(1, len(dl_tr)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = torch.nn.BCEWithLogitsLoss()
    best, history = math.inf, []
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        tl, n = 0.0, 0
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = bce(logits, y) + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(x)
            n += len(x)
        model.eval()
        vl, vn = 0.0, 0
        tp = np.zeros(k)
        fp = np.zeros(k)
        fn = np.zeros(k)
        with torch.inference_mode():
            for x, y in dl_va:
                x = x.to(dev).to(memory_format=torch.channels_last)
                y = y.to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(x)
                vl += float(bce(logits.float(), y) + dice_loss(logits.float(), y)) * len(x)
                vn += len(x)
                p = (logits.float().sigmoid() > 0.5).float()
                tp += (p * y).sum(dim=(0, 2, 3)).cpu().numpy()
                fp += (p * (1 - y)).sum(dim=(0, 2, 3)).cpu().numpy()
                fn += ((1 - p) * y).sum(dim=(0, 2, 3)).cpu().numpy()
        vl /= max(vn, 1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        rec = {"epoch": ep + 1, "train_loss": round(tl / max(n, 1), 4), "val_loss": round(vl, 4),
               "pixel_f1": {cls: round(float(v), 3) for cls, v in zip(meta["classes"], f1)},
               "seconds": round(time.perf_counter() - t0, 1)}
        history.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
        if vl < best:
            best = vl
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "encoder": encoder, "in_channels": c,
                        "classes": meta["classes"], "markup": meta["markup"], "res": meta["res"],
                        "tile": meta["tile"], "li_tags": meta.get("li_tags"), "epoch": ep + 1,
                        "val_loss": vl}, out)
    return {"best_val_loss": best, "history": history, "out": out,
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = train(args.cache, args.out, args.encoder, args.epochs, args.batch, args.lr, args.val_families,
                args.workers, not args.no_pretrained, args.limit, not args.no_amp)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
