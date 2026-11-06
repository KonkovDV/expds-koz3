#!/usr/bin/env python3
"""Ядро Kaggle: YOLO26 на окнах спутника (план docs/plans/OPTICS_DETECTOR_PLAN_20261105.md, §3.3–3.6).

    python make_yolo_kernel.py --slug koz3-spor-yolo-f0 --folds 0 --epochs 40 --out BUILD [--model yolo26s.pt]
                               [--imgsz 1024] [--batch 16] [--fraction 1.0]

Вход — приватный датасет `dmitrijkv/koz3-spor-yolo` (выход export_spor.py). Kaggle распаковывает zip без папок,
поэтому ядро само раскладывает файлы: *.jpg → ds/images, *.txt окон → ds/labels, списки и yaml — с путями ds.
Для каждого фолда k из --folds: обучение на train_f{k} (без семей фолда k), валидация на val_f{k}; затем OOF —
предсказания best.pt на окнах val_f{k} (conf ≥ 0,01) в oof_f{k}.jsonl: окно, xyxy в пикселях окна, conf, cls.
Модель сдачи по уроку EN — фолдовая модель с порогами её же OOF (§3.6), а не пул фолдов с NMS-max.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OWNER = "dmitrijkv"
DATASET = f"{OWNER}/koz3-spor-yolo"

SCRIPT = r'''# YOLO26 на окнах спутника SpOr (план OPTICS_DETECTOR_PLAN_20261105), {cfg}
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path
CFG = json.loads({cfg!r})
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4.0"], check=True)
idx = sorted(glob.glob("/kaggle/input/**/index.csv", recursive=True))
assert len(idx) == 1, idx
src = Path(idx[0]).parent
inp = Path("/kaggle/input")
ds = Path("/kaggle/working/ds")
(ds / "images").mkdir(parents=True, exist_ok=True)
(ds / "labels").mkdir(parents=True, exist_ok=True)
n_img = n_lab = 0
for p in inp.rglob("*.jpg"):
    d = ds / "images" / p.name
    if not d.exists():
        os.symlink(p, d); n_img += 1
for p in inp.rglob("*.txt"):
    if p.name.startswith(("train_", "val_")):
        shutil.copy(p, ds / p.name)
    else:
        d = ds / "labels" / p.name
        if not d.exists():
            os.symlink(p, d); n_lab += 1
for p in inp.rglob("data_f*.yaml"):
    (ds / p.name).write_text(p.read_text(encoding="utf-8").replace("path: .", f"path: {{ds}}"), encoding="utf-8")
print("окон", n_img, "разметок", n_lab, flush=True)
from ultralytics import YOLO
import torch
dev = list(range(torch.cuda.device_count())) or "cpu"
for k in CFG["folds"]:
    tk = time.time()
    m = YOLO(CFG["model"])
    base = dict(data=str(ds / f"data_f{{k}}.yaml"), imgsz=CFG["imgsz"], epochs=CFG["epochs"], batch=CFG["batch"], device=dev,
                workers=4, project="/kaggle/working/runs", name=f"f{{k}}", exist_ok=True, seed=0, plots=False,
                fraction=CFG["fraction"], flipud=0.5, fliplr=0.5)
    if CFG["recipe"] == "gigaflex":   # OSINT: Strekalov/arhipelag train.py (§3.9 плана), flipud — наше (снимок сверху)
        base.update(optimizer="AdamW", lr0=2.41e-04, lrf=0.038, weight_decay=0.007489, momentum=0.96, warmup_epochs=3,
                    warmup_momentum=0.93, cos_lr=False, box=29.7, cls=3.5, dfl=3.9, hsv_h=0.4, hsv_s=0.3, hsv_v=0.9,
                    translate=0.05, mixup=0.2, cutmix=0.5, scale=0.312, degrees=35, mosaic=0.8, bgr=0.5,
                    close_mosaic=max(1, CFG["epochs"] // 3), patience=300)
    else:                             # базовый рецепт ultralytics + D2 и умеренный масштаб (§3.4)
        base.update(cos_lr=True, close_mosaic=5, degrees=0.0, scale=0.25, mosaic=1.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
                    patience=0)
    m.train(**base)
    best = Path(f"/kaggle/working/runs/f{{k}}/weights/best.pt")
    shutil.copy(best, f"/kaggle/working/yolo_spor_f{{k}}.pt")
    val = [str(ds / l.strip()[2:]) if l.strip().startswith("./") else l.strip()
           for l in open(ds / f"val_f{{k}}.txt", encoding="utf-8") if l.strip()]
    mb = YOLO(str(best))
    with open(f"/kaggle/working/oof_f{{k}}.jsonl", "w", encoding="utf-8") as fo:
        for i in range(0, len(val), 64):
            for r in mb.predict(source=val[i:i + 64], imgsz=CFG["imgsz"], conf=0.01, iou=0.6, device=dev[0] if dev != "cpu" else "cpu",
                                verbose=False, batch=16):
                b = r.boxes
                fo.write(json.dumps({{"tile": Path(r.path).stem, "xyxy": b.xyxy.cpu().numpy().round(1).tolist(),
                                      "conf": b.conf.cpu().numpy().round(5).tolist(),
                                      "cls": b.cls.cpu().numpy().astype(int).tolist()}}) + "\n")
    print(f"фолд {{k}}: обучение+OOF {{time.time() - tk:.0f}} с, окон OOF {{len(val)}}", flush=True)
print("ИТОГ: OK за", round(time.time() - t0), "с", flush=True)
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--recipe", default="default", choices=("default", "gigaflex"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path("/workspace/c12/BUILD/koz3-li-c12-in-a/kernel-metadata.json"))
    a = ap.parse_args()
    cfg = {"folds": [int(x) for x in a.folds.split(",")], "epochs": a.epochs, "model": a.model, "imgsz": a.imgsz,
           "batch": a.batch, "fraction": a.fraction, "recipe": a.recipe}
    kd = a.out / a.slug
    kd.mkdir(parents=True, exist_ok=True)
    text = SCRIPT.format(cfg=json.dumps(cfg))
    compile(text, a.slug, "exec")
    (kd / f"{a.slug}.py").write_text(text, encoding="utf-8")
    meta = json.loads(a.template.read_text(encoding="utf-8"))
    meta.update(id=f"{OWNER}/{a.slug}", title=a.slug, code_file=f"{a.slug}.py", is_private=True, enable_gpu=True,
                enable_internet=True, dataset_sources=[DATASET], kernel_sources=[])
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"kernel": str(kd), **cfg}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
