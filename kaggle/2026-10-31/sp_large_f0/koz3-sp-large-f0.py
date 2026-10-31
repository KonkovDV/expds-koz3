#!/usr/bin/env python3
"""KOZ3 sp_large fold0 (PLAN_SP_LARGE_ONE_GPU_20261031, Elena 31.10).
Same net as submitted spor_focal_r18: ResNet-18 U-Net, centers, focal, ImageNet init (smp).
ONLY change: peak loss weight x4 for damaged kurgans with GT area >= 1000 m2, x2 for intact >= 1000 m2.
Fold 0 only (family 'Курганы', 34 regions). Optional control (same recipe, weight 1) only if deadline allows.
No credentials, no S3. Deadline ceiling SPL_DEADLINE_H."""
import os, sys, subprocess, time, json, tarfile, shutil, importlib, importlib.metadata
from pathlib import Path
T0 = time.time()
DEADLINE_H = float(os.environ.get("SPL_DEADLINE_H", "4.0"))
RUN_CONTROL = os.environ.get("SPL_CONTROL", "1") == "1"
OUT = Path("/kaggle/working"); W = Path("/tmp/spl")
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)
def left_h(): return DEADLINE_H - (time.time() - T0) / 3600
(OUT / "STATUS.txt").write_text("RUNNING\n")
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
if "GPU" not in smi:
    (OUT / "STATUS.txt").write_text("FAILED no GPU\n"); raise SystemExit("no GPU")
log("gpu", smi.strip())
import torch
cons = "/tmp/cons.txt"
Path(cons).write_text("".join(f"{p}=={importlib.metadata.version(p)}\n" for p in ("torch", "torchvision", "numpy")))
need = []
for mod, pin in {"segmentation_models_pytorch": "segmentation-models-pytorch==0.6.0", "timm": "timm==1.0.29",
                 "rasterio": "rasterio==1.4.3", "pyproj": "pyproj==3.7.1", "shapely": "shapely==2.1.1",
                 "scipy": "scipy==1.16.3", "laspy": "laspy==2.7.0"}.items():
    try: importlib.import_module(mod)
    except Exception: need.append(pin)
if need:
    log("pip", need)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", "-c", cons, *need])
    if r.returncode: (OUT / "STATUS.txt").write_text("FAILED pip\n"); raise SystemExit("pip failed")
INP = next(p for p in Path("/kaggle/input").rglob("code.tar")).parent
log("input", INP)
W.mkdir(parents=True, exist_ok=True)
for t in ("code.tar", "cache.tar", "root_fold0.tar"):
    dst = W / ("root" if t.startswith("root") else "")
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(INP / t) as tf: tf.extractall(dst)
log("extracted", sorted(os.listdir(W)), len(os.listdir(W / "root")), "regions")
COMMON = ["--markup", "sp_or", "--root", str(W / "root"), "--cache", str(W / "cache_spor_r1_t128_s25"),
          "--folds", "5", "--fold-indices", "0", "--epochs", "30", "--batch", "32",
          "--classes", "kurgany_povrezhdennye", "kurgany_tselye",
          "--target", "centers", "--center-loss", "focal", "--center-sigma", "0",
          "--center-radius", "4", "--center-min-dist", "12", "--config", str(W / "configs" / "default.toml")]
JOBS = [("sp_large", ["--large-kurgan-weight", "2", "--large-damaged-weight", "4", "--large-area-m2", "1000"])]
if RUN_CONTROL: JOBS.append(("control", []))
env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(W)}
for tag, extra in JOBS:
    if tag == "control" and left_h() < 1.2:
        log("skip control: deadline"); continue
    work = W / f"work_{tag}"; od = OUT / tag; od.mkdir(exist_ok=True)
    cmd = [sys.executable, "-X", "utf8", "scripts/nn_oof.py", *COMMON, "--work", str(work),
           "--out", str(od / f"{tag}_fold0_oof.geojson"), *extra]
    log("run", tag, " ".join(cmd[3:]))
    budget = max(600, left_h() * 3600 - 300)
    with open(od / f"{tag}.log", "w") as fh:
        try: rc = subprocess.run(cmd, cwd=W, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=budget).returncode
        except subprocess.TimeoutExpired: rc = "timeout"
    log(tag, "rc", rc)
    for f in ("fold0/fold_report.json", "fold0/done.json", "fold0/pred.geojson", "fold0/sp_or.pt"):
        p = work / f
        if p.exists(): shutil.copy(p, od / Path(f).name)
    if tag == "sp_large" and rc != 0:
        (OUT / "STATUS.txt").write_text(f"FAILED sp_large rc={rc}\n"); raise SystemExit(1)
(OUT / "STATUS.txt").write_text(f"DONE {time.time()-T0:.0f}s\n"); log("DONE")
