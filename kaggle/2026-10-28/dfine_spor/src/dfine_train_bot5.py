# Адаптер тренера бота 5 под хук dfine_main.py: train(model, batches, cfg, device, log) -> model.
# Вендорено из /workspace/koz3_shared/bot-5/repo, ветка bot5-dfine-kaggle, коммит 0256251,
#   scripts/det_train_hf.py (git blob 45a67faee1823acde38f930e5c6f3a7b6e3363b3, sha256 78cab4ac41081c2d…):
#   jitter() — дословно (стр. 49–80); цикл обучения main() (AdamW 2 группы, OneCycle, GradScaler, autocast, clip) — дословно
#   по логике, переписан как функция. Отличия адаптера (документированы в bot-9/DFINE.md):
#   * батчи приходят из нашего конвейера (pixel_values 0..1 без нормализации = RTDetrImageProcessor do_rescale, do_normalize=false;
#     labels class_labels + boxes cxcywh норм. — тот же формат, что даёт процессор), поэтому AutoImageProcessor и COCO-датасет
#     бота 5 не нужны;
#   * добавлены: стоп по бюджету времени/дедлайну (cfg budget_s/stop_ts), пропуск нефинитного loss (≤50 подряд), отчёт
#     model.train_report в формате встроенного тренера; fp16 включается на cuda (как --fp16 бота 5).
import math, random, time
import numpy as np
from PIL import Image, ImageFilter

def jitter(img: np.ndarray, boxes: np.ndarray, rng: random.Random, strong: bool):
    """[бот 5, дословно] img HxWx3 uint8, boxes Nx4 xywh в пикселях."""
    h, w = img.shape[:2]
    x = img.astype(np.float32)
    b = boxes.astype(np.float32).copy()
    if rng.random() < 0.5:  # отражение по горизонтали
        x = x[:, ::-1]
        if len(b):
            b[:, 0] = w - b[:, 0] - b[:, 2]
    k = rng.randrange(4)
    for _ in range(k):  # поворот на 90° против часовой: (x, y) -> (y, W - x)
        hh, ww = x.shape[:2]
        x = np.rot90(x)
        if len(b):
            nx, ny = b[:, 1].copy(), ww - b[:, 0] - b[:, 2]
            b = np.stack([nx, ny, b[:, 3], b[:, 2]], 1)
    if rng.random() < 0.8:
        x = x * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.1:
        x = np.repeat(x.mean(2, keepdims=True), 3, 2)
    out = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    if rng.random() < 0.2:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if strong and rng.random() < 0.3:
        f = rng.uniform(0.4, 0.8)
        sz = out.size
        out = out.resize((max(8, int(sz[0] * f)), max(8, int(sz[1] * f))), Image.BILINEAR).resize(sz, Image.BILINEAR)
    return np.asarray(out), b

def train(model, batches, cfg, device, log):
    import torch
    dev = device; tag = cfg.get("tag", "")
    iters = int(cfg["iters"]); lr, mult, wd = cfg["lr"], cfg["bb_mult"], cfg["wd"]
    fp16 = dev.type == "cuda"
    # --- бот 5: оптимизатор и расписание ---
    bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    lrs = [lr * mult, lr]
    opt = torch.optim.AdamW([{"params": bb, "lr": lrs[0]}, {"params": rest, "lr": lrs[1]}], weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lrs, total_steps=max(2, iters), pct_start=cfg["pct"])
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    model.train()
    t = time.time(); hist = []; tm = []; rep = []; it = 0; bad = 0
    log_every = 1 if cfg.get("mode") == "smoke" else 100
    budget = cfg.get("budget_s"); stop_ts = cfg.get("stop_ts")
    while it < iters:
        pv, lab = next(batches); ts = time.time()
        with torch.autocast(dev.type, enabled=fp16):
            o = model(pixel_values=pv.to(dev), labels=[{k: v.to(dev) for k, v in l.items()} for l in lab])
        if not torch.isfinite(o.loss.float()):  # [адаптер] защита, у бота 5 её нет
            bad += 1; opt.zero_grad(set_to_none=True)
            if bad > 50: raise RuntimeError("loss NaN 50 раз подряд — перезапустить с DFINE_NUM_DN=0")
            continue
        bad = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(o.loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        scaler.step(opt); scaler.update()
        if it < sch.total_steps - 1: sch.step()
        if fp16: torch.cuda.synchronize(dev)
        hist.append(float(o.loss.detach())); tm.append(time.time() - ts); it += 1
        if it % log_every == 0 or it == iters:
            rep.append({"it": it, "loss": round(float(np.mean(hist[-log_every:])), 4), "s_it": round(float(np.median(tm[-50:])), 4)})
            log(tag, "[bot5]", rep[-1])
        now = time.time()
        if (budget and now - t > budget) or (stop_ts and now > stop_ts):  # [адаптер] стоп по времени
            log(tag, "[bot5] СТОП по времени на ит.", it); break
    model.train_report = {"trainer": "bot5@0256251", "iters": it, "planned": iters,
                          "s_it_median": float(np.median(tm[min(5, len(tm) - 1):])) if tm else None,
                          "hist": rep, "sec": round(time.time() - t), "loss_first": hist[0] if hist else None,
                          "loss_last": hist[-1] if hist else None}
    return model
