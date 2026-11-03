#!/usr/bin/env python3
"""koz3-sat-cl: U-Net (smp, ConvNeXt-T ImageNet) for gorodishcha + arkhitektury masks on SpOr, 5-fold OOF.

Why (docs/WHY_LOW_SCORE_2026-11-03.md): optical hillforts are mostly "wrong class" (+0.042) and optical
architecture mostly "no candidate" (+0.037); the shipped satellite models for both are ResNet-18 U-Nets.
Train tiles: W1 sat2 grid (0.5 m, 512 px, folds pinned like spor_focal) from dmitrijkv/koz3-w1-sat2-f14-data;
targets rasterized from gt_train_v2.geojson. OOF: full SpOr rasters of the held-out fold
(dmitrijkv/koz3-w2-dfine2-raw-a/-b, ASCII names mapped back by manifest_*.json), sliding window,
components -> EPSG:3857 polygons, confidence = max prob, threshold_group "sp_cl" (thresholds are chosen
honestly by the gate). No secrets, no S3; weights and OOF go to /kaggle/working.

Inputs are found under KIN (default /kaggle/input) by file name.
Env: FOLDS (default "0,1,2,3,4"), ITERS (2500), BATCH (8), ENC (tu-convnext_tiny), DEADLINE_H (8), SMOKE (0).
"""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
CLASSES = ("gorodishcha", "arkhitektury")
IN = Path(os.environ.get("KIN", "/kaggle/input"))
OUT = Path(os.environ.get("KOUT", "/kaggle/working"))
# Kaggle mount points differ between images (/kaggle/input/<slug> or deeper), so inputs are found by name:
# grid fold dirs = parents of tiles.csv named f0..f4; raw dirs = parents of manifest_*.json (as the D-FINE-2 kernel does).
ENC = os.environ.get("ENC", "tu-convnext_tiny")
ITERS = int(os.environ.get("ITERS", "2500"))
BATCH = int(os.environ.get("BATCH", "8"))
DEADLINE = T0 + float(os.environ.get("DEADLINE_H", "8")) * 3600
SMOKE = os.environ.get("SMOKE", "0") == "1"
FLOOR = {"gorodishcha": 0.15, "arkhitektury": 0.15}   # generation floor; the gate sweeps thresholds above it
TILE, OVL, STEP_M = 512, 128, 0.5


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def pip():
    pk = ["segmentation-models-pytorch==0.5.0", "timm==1.0.30", "rasterio", "shapely>=2", "pyproj"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pk], check=False)


def raw_dirs():
    return sorted({m.parent for m in IN.rglob("manifest_*.json")})


def load_gt():
    gts = sorted(IN.rglob("gt_train_v2.geojson"))
    if not gts:
        raise FileNotFoundError(f"gt_train_v2.geojson not found under {IN}")
    gt = gts[0]
    feats = json.loads(gt.read_text(encoding="utf-8"))["features"]
    by = {}
    for f in feats:
        pr = f.get("properties") or {}
        if pr.get("markup_type") == "sp_or" and pr.get("class_name") in CLASSES:
            by.setdefault(pr["region_name"], []).append(f)
    return by


def grid_tiles():
    rows = []
    fds = sorted({p.parent for p in IN.rglob("tiles.csv") if p.parent.name in ("f0", "f1", "f2", "f3", "f4")}, key=lambda d: d.name)
    if not fds:
        raise FileNotFoundError(f"grid tiles.csv (f0..f4) not found under {IN}")
    for fd in fds:
        for r in csv.DictReader(open(fd / "tiles.csv", encoding="utf-8")):
            r["dir"] = str(fd); rows.append(r)
    return rows


class Tiles:
    """Grid tiles + GT masks rasterized on the fly in each tile's UTM frame."""

    def __init__(self, rows, gt_by):
        import numpy as np
        from pyproj import Transformer
        from shapely.geometry import shape
        from shapely.ops import transform as stx
        self.rows, self.np, self.sh = rows, np, {}
        self.geo = {}
        for r in rows:
            k = (r["region"], r["crs"])
            if k in self.geo:
                continue
            tf = Transformer.from_crs(3857, r["crs"], always_xy=True)
            self.geo[k] = [(CLASSES.index(f["properties"]["class_name"]), stx(tf.transform, shape(f["geometry"])))
                           for f in gt_by.get(r["region"], [])]
        self.pos = np.array([self._has(r) for r in rows], bool)

    def _has(self, r):
        from shapely.geometry import box
        L, T, s = float(r["left"]), float(r["top"]), float(r["step"])
        b = box(L, T - TILE * s, L + TILE * s, T)
        return any(g.intersects(b) for _, g in self.geo[(r["region"], r["crs"])])

    def image(self, r):
        k = (r["dir"], r["shard"])
        if k not in self.sh:
            self.sh[k] = self.np.load(Path(r["dir"]) / f"images_{int(r['shard']):04d}.npy", mmap_mode="r")
        return self.np.asarray(self.sh[k][int(r["idx"])])

    def mask(self, r):
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        np = self.np
        L, T, s = float(r["left"]), float(r["top"]), float(r["step"])
        m = np.zeros((len(CLASSES), TILE, TILE), np.uint8)
        tr = from_origin(L, T, s, s)
        for c in range(len(CLASSES)):
            gs = [g for ci, g in self.geo[(r["region"], r["crs"])] if ci == c and not g.is_empty]
            if gs:
                m[c] = rasterize([(g, 1) for g in gs], out_shape=(TILE, TILE), transform=tr, fill=0, all_touched=True)
        return m


def train_fold(k, rows, gt_by, dev):
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F
    tr_rows = [r for r in rows if int(r["fold"]) != k]
    ds = Tiles(tr_rows, gt_by)
    rng = np.random.default_rng(1000 + k)
    w = np.where(ds.pos, 4.0, 1.0); w /= w.sum()
    log(f"f{k}: train tiles {len(tr_rows)}, with gor/arch {int(ds.pos.sum())}")
    model = smp.Unet(encoder_name=ENC, encoder_weights="imagenet", classes=len(CLASSES)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=ITERS, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    model.train()
    for it in range(ITERS):
        idx = rng.choice(len(tr_rows), BATCH, p=w)
        xs, ys = [], []
        for i in idx:
            x, y = ds.image(tr_rows[i]).astype(np.float32) / 255.0, ds.mask(tr_rows[i]).astype(np.float32)
            kk, fl = rng.integers(4), rng.random() < 0.5
            x, y = np.rot90(x, kk, (1, 2)), np.rot90(y, kk, (1, 2))
            if fl:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06)
            xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
        x = (torch.from_numpy(np.stack(xs)).to(dev) - mean) / std
        y = torch.from_numpy(np.stack(ys)).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            lo = model(x)
        lo = lo.float()
        bce = F.binary_cross_entropy_with_logits(lo, y, pos_weight=torch.tensor([3.0, 3.0], device=dev).view(1, 2, 1, 1))
        p = torch.sigmoid(lo)
        dice = 1 - ((2 * (p * y).sum((0, 2, 3)) + 1) / (p.sum((0, 2, 3)) + y.sum((0, 2, 3)) + 1)).mean()
        loss = bce + dice
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 200 == 0 or it == ITERS - 1:
            log(f"f{k} it {it} loss {loss.item():.4f} (bce {bce.item():.4f} dice {dice.item():.4f})")
        if time.time() > DEADLINE - 3600:
            log(f"f{k}: deadline, stop training at it {it}"); break
    model.eval()
    torch.save({"state_dict": model.state_dict(), "enc": ENC, "classes": CLASSES, "fold": k, "iters": it + 1},
               OUT / f"cl_f{k}.pt")
    return model


def raw_index():
    idx = {}
    for d in raw_dirs():
        for man in d.glob("manifest_*.json"):
            for a, v in json.loads(man.read_text(encoding="utf-8")).items():
                if (d / a).exists():
                    idx.setdefault(v["region"], []).append((d / a, v["file"]))
    return idx


def infer_region(model, dev, region, files, k):
    import numpy as np
    import rasterio
    import torch
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.features import shapes
    from scipy import ndimage
    from shapely.geometry import shape
    from shapely.ops import transform as stx
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for path, orig in files:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                log(f"{region} {orig}: no CRS, skip"); continue
            res = abs(ds.transform.a)
            step = max(STEP_M, res / 2.0)                 # grid: max_upsample 2
            H, W = max(1, int(round(ds.height * res / step))), max(1, int(round(ds.width * res / step)))
            arr = ds.read(indexes=[1, 2, 3] if ds.count >= 3 else [1, 1, 1], out_shape=(3, H, W),
                          resampling=Resampling.average if step > res else Resampling.bilinear)
            tr = ds.transform * ds.transform.scale(ds.width / W, ds.height / H)
            crs = ds.crs
        if arr.dtype != np.uint8:
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
        x_all = arr.astype(np.float32) / 255.0
        prob = np.zeros((len(CLASSES), H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        ys = list(range(0, max(H - TILE, 0) + 1, TILE - OVL)) or [0]
        xs = list(range(0, max(W - TILE, 0) + 1, TILE - OVL)) or [0]
        if ys[-1] + TILE < H: ys.append(H - TILE)
        if xs[-1] + TILE < W: xs.append(W - TILE)
        with torch.no_grad():
            for y0 in ys:
                for x0 in xs:
                    y0c, x0c = max(0, y0), max(0, x0)
                    win = x_all[:, y0c:y0c + TILE, x0c:x0c + TILE]
                    h, w = win.shape[1:]
                    pad = np.zeros((3, TILE, TILE), np.float32); pad[:, :h, :w] = win
                    t = (torch.from_numpy(pad)[None].to(dev) - mean) / std
                    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                        p = torch.sigmoid(model(t).float())[0].cpu().numpy()
                    prob[:, y0c:y0c + h, x0c:x0c + w] += p[:, :h, :w]; cnt[y0c:y0c + h, x0c:x0c + w] += 1
        prob /= np.maximum(cnt, 1)
        to3857 = Transformer.from_crs(crs, 3857, always_xy=True)
        for ci, cls in enumerate(CLASSES):
            m = prob[ci] >= FLOOR[cls]
            if not m.any():
                continue
            lab, n = ndimage.label(m)
            if n == 0:
                continue
            mx = ndimage.maximum(prob[ci], lab, index=np.arange(1, n + 1))
            for geom, v in shapes(lab.astype(np.int32), mask=m, transform=tr):
                v = int(v)
                if v <= 0:
                    continue
                g = shape(geom)
                if g.area < 4 * step * step:
                    continue
                g = stx(to3857.transform, g.simplify(step / 2))
                if g.is_empty or not g.is_valid:
                    g = g.buffer(0)
                if g.is_empty:
                    continue
                if g.geom_type == "MultiPolygon":
                    g = max(g.geoms, key=lambda q: q.area)
                out.append({"type": "Feature", "geometry": g.__geo_interface__,
                            "properties": {"class_name": cls, "confidence": round(float(mx[v - 1]), 6),
                                           "region_name": region, "markup_type": "sp_or", "threshold_group": "sp_cl",
                                           "crs": "EPSG:3857", "fold": k, "source": orig}})
    return out


def run_fold(k, dev_name):
    import torch
    dev = torch.device(dev_name)
    gt_by = load_gt(); rows = grid_tiles()
    if SMOKE:
        rows = rows[:64]
    model = train_fold(k, rows, gt_by, dev)
    regions = sorted({r["region"] for r in grid_tiles() if int(r["fold"]) == k})
    idx = raw_index(); feats = []
    for i, reg in enumerate(regions[:2] if SMOKE else regions):
        if time.time() > DEADLINE - 600:
            log(f"f{k}: deadline in inference, {i}/{len(regions)} regions"); break
        feats += infer_region(model, dev, reg, idx.get(reg, []), k)
    (OUT / f"oof_cl_f{k}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                                         ensure_ascii=False), encoding="utf-8")
    log(f"f{k}: OOF {len(feats)} features over {len(regions)} regions")
    return {"fold": k, "regions": len(regions), "features": len(feats)}


def main():
    if "--fold" in sys.argv:
        k = int(sys.argv[sys.argv.index("--fold") + 1]); dev = sys.argv[sys.argv.index("--dev") + 1]
        rep = run_fold(k, dev)
        (OUT / f"report_f{k}.json").write_text(json.dumps(rep), encoding="utf-8")
        return
    pip()
    import torch
    folds = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
    ngpu = max(1, torch.cuda.device_count())
    devs = [f"cuda:{i}" for i in range(ngpu)] if torch.cuda.is_available() else ["cpu"]
    log(f"folds {folds}, devices {devs}, enc {ENC}, iters {ITERS}, batch {BATCH}")
    queue = list(folds); procs = {}
    while queue or procs:
        for d in devs:
            if d not in procs and queue:
                k = queue.pop(0)
                procs[d] = (k, subprocess.Popen([sys.executable, __file__, "--fold", str(k), "--dev", d]))
        time.sleep(10)
        for d, (k, p) in list(procs.items()):
            if p.poll() is not None:
                log(f"fold {k} on {d} exited {p.returncode}"); del procs[d]
    feats = []
    for k in folds:
        p = OUT / f"oof_cl_f{k}.geojson"
        if p.exists():
            feats += json.loads(p.read_text(encoding="utf-8"))["features"]
    (OUT / "oof_cl_spor.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                                        ensure_ascii=False), encoding="utf-8")
    reps = [json.loads((OUT / f"report_f{k}.json").read_text()) for k in folds if (OUT / f"report_f{k}.json").exists()]
    (OUT / "cl_report.json").write_text(json.dumps({"enc": ENC, "iters": ITERS, "batch": BATCH, "folds": reps,
                                                    "sec": round(time.time() - T0)}, indent=1), encoding="utf-8")
    log("DONE", len(feats), "OOF features")


if __name__ == "__main__":
    main()
