#!/usr/bin/env python3
"""Замер скорости обучения и инференса сегментационных моделей на вашей видеокарте.

Работает на синтетических данных: конкурсный датасет не нужен и не читается.
Цель — за 10–15 минут получить реальные цифры для GTX 1650 (или любой другой
карты) и посчитать, сколько часов займёт обучение по модальностям.

Установка (Windows или Linux, драйвер NVIDIA >= 525):
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    pip install segmentation-models-pytorch

Запуск:
    python bench_gpu.py                 # стандартный набор конфигураций
    python bench_gpu.py --tiles 30000 --epochs 20   # свой объём данных
    python bench_gpu.py --quick         # быстрая проверка, ~1 минута

Результат печатается таблицей и сохраняется в bench_result.json — его можно
прислать для расчёта бюджета (в нём нет конкурсных данных).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass

import torch
import segmentation_models_pytorch as smp


@dataclass
class Config:
    """Одна конфигурация замера."""
    encoder: str
    size: int
    batch: int
    in_ch: int = 3
    classes: int = 4


@dataclass
class Result:
    """Итог замера одной конфигурации."""
    encoder: str
    size: int
    batch: int
    train_img_s: float | None
    infer_img_s: float | None
    peak_mem_gb: float | None
    hours_per_model: float | None
    note: str = ""


DEFAULT_CONFIGS = [
    Config("resnet18", 256, 16),
    Config("resnet34", 256, 16),
    Config("efficientnet-b0", 256, 16),
    Config("mobilenet_v2", 256, 16),
    Config("resnet18", 384, 8),
    Config("resnet18", 512, 4),
]


def build_model(cfg: Config) -> torch.nn.Module:
    """U-Net из segmentation_models_pytorch без загрузки весов (скорость та же)."""
    return smp.Unet(encoder_name=cfg.encoder, encoder_weights=None,
                    in_channels=cfg.in_ch, classes=cfg.classes)


def measure(cfg: Config, device: torch.device, steps: int, warmup: int,
            amp: bool, tiles: int, epochs: int) -> Result:
    """Меряет обучение (forward+backward+step) и инференс на случайных тензорах."""
    use_cuda = device.type == "cuda"
    try:
        fmt = torch.channels_last if amp and use_cuda else torch.contiguous_format   # как в expds.train
        model = build_model(cfg).to(device).to(memory_format=fmt)
    except Exception as exc:  # энкодер может отсутствовать в вашей версии smp
        return Result(cfg.encoder, cfg.size, cfg.batch, None, None, None, None,
                      f"энкодер недоступен: {type(exc).__name__}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and use_cuda)
    x = torch.randn(cfg.batch, cfg.in_ch, cfg.size, cfg.size, device=device)
    x = x.contiguous(memory_format=fmt)
    y = (torch.rand(cfg.batch, cfg.classes, cfg.size, cfg.size, device=device) > 0.95).float()

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        for i in range(warmup + steps):
            if i == warmup:
                sync()
                t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and use_cuda):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sync()
        train_ips = cfg.batch * steps / (time.perf_counter() - t0)

        model.eval()
        with torch.inference_mode():
            for i in range(warmup + steps):
                if i == warmup:
                    sync()
                    t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and use_cuda):
                    model(x)
            sync()
            infer_ips = cfg.batch * steps / (time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if use_cuda else None
        hours = tiles * epochs / train_ips / 3600
        return Result(cfg.encoder, cfg.size, cfg.batch, round(train_ips, 1),
                      round(infer_ips, 1), None if peak is None else round(peak, 2),
                      round(hours, 1))
    except torch.cuda.OutOfMemoryError:
        return Result(cfg.encoder, cfg.size, cfg.batch, None, None, None, None,
                      "не хватило памяти: уменьшите --batch")
    finally:
        del model, opt
        if use_cuda:
            torch.cuda.empty_cache()


def main() -> None:
    """Разбор аргументов, прогон конфигураций, печать и сохранение результатов."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=30000,
                    help="тайлов в эпохе для пересчёта в часы (по умолчанию 30000)")
    ap.add_argument("--epochs", type=int, default=20, help="эпох на модель")
    ap.add_argument("--steps", type=int, default=40, help="замеряемых шагов")
    ap.add_argument("--warmup", type=int, default=8, help="шагов прогрева")
    ap.add_argument("--no-amp", action="store_true", help="отключить fp16 (AMP)")
    ap.add_argument("--quick", action="store_true", help="только resnet18@256, мало шагов")
    ap.add_argument("--cpu", action="store_true", help="принудительно CPU (для проверки)")
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = not args.no_amp
    if amp and device.type == "cuda" and "GTX 16" in torch.cuda.get_device_name(0):
        print("GTX 16xx: FP16 в cuDNN даёт NaN, замер в FP32 (как при обучении).")
        amp = False
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "smp": smp.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "capability": list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        if device.type == "cuda" else None,
        "amp_fp16": amp and device.type == "cuda",
        "tiles": args.tiles, "epochs": args.epochs,
    }
    if device.type == "cpu":
        print("ВНИМАНИЕ: CUDA не найдена — замер идёт на CPU, цифры не про видеокарту.")
    torch.backends.cudnn.benchmark = True

    configs = [Config("resnet18", 256, 16)] if args.quick else DEFAULT_CONFIGS
    steps = 10 if args.quick else args.steps
    warmup = 3 if args.quick else args.warmup

    print(f"Устройство: {info['device']}  VRAM: {info['vram_gb']} ГБ  AMP: {info['amp_fp16']}")
    print(f"Пересчёт: {args.tiles} тайлов × {args.epochs} эпох на одну модель\n")
    print(f"{'энкодер':18s} {'тайл':>5s} {'батч':>5s} {'обуч, img/s':>12s} "
          f"{'инфер, img/s':>13s} {'пик VRAM':>9s} {'часов/модель':>13s}")
    results = []

    def fmt(v):
        return "—" if v is None else str(v)

    for cfg in configs:
        r = measure(cfg, device, steps, warmup, amp, args.tiles, args.epochs)
        results.append(r)
        print(f"{r.encoder:18s} {r.size:5d} {r.batch:5d} {fmt(r.train_img_s):>12s} "
              f"{fmt(r.infer_img_s):>13s} {fmt(r.peak_mem_gb):>9s} "
              f"{fmt(r.hours_per_model):>13s} {r.note}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"info": info, "results": [asdict(r) for r in results]}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
