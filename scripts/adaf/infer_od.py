#!/usr/bin/env python3
"""Zero-shot ADAF Faster R-CNN inference on SLRM tiles (own re-implementation, no aitlas).

Model: torchvision fasterrcnn_resnet50_fpn_v2, num_classes=2 (background + 1), as in ADAF/aitlas
(Apache-2.0). Weights: ADAF OD_* (Zenodo 10.5281/zenodo.15848663, CC-BY-SA-4.0).
Input like ADAF detect_objects_v2: float [0,1] SLRM repeated to 3 channels, no extra scaling;
torchvision's GeneralizedRCNNTransform does ImageNet mean/std and resize (min_size 800).
Tiles: --tile px (ADAF training patches 512 px @0.5 m) with --stride; NMS across tiles.
Output: <out>/<model>/<region>.json  [{x,y (UTM centre), w,h (m), score}]
"""
from __future__ import annotations
import argparse, glob, json, time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


def load_model(path: str, min_size: int) -> torch.nn.Module:
    m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=2,
                                   min_size=min_size, max_size=max(1333, min_size))
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, missing[:5]
    m.eval()
    return m


@torch.no_grad()
def run_region(m, z, tile, stride, min_valid):
    u8, valid = z["slrm_u8"], z["valid"]
    H, W = u8.shape
    boxes, scores = [], []
    ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < H: ys.append(max(H - tile, 0))
    if xs[-1] + tile < W: xs.append(max(W - tile, 0))
    n = 0
    for y in ys:
        for x in xs:
            v = valid[y:y + tile, x:x + tile]
            if v.mean() < min_valid:
                continue
            a = np.zeros((tile, tile), np.float32)
            p = u8[y:y + tile, x:x + tile]
            a[:p.shape[0], :p.shape[1]] = p / 255.0
            t = torch.from_numpy(a)[None].repeat(3, 1, 1)
            o = m([t])[0]
            b = o["boxes"].numpy() + np.array([x, y, x, y], np.float32)
            boxes.append(b); scores.append(o["scores"].numpy()); n += 1
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0
    b = torch.from_numpy(np.concatenate(boxes)); s = torch.from_numpy(np.concatenate(scores))
    keep = torchvision.ops.nms(b, s, 0.3).numpy()
    return b.numpy()[keep], s.numpy()[keep], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slrm", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--min-size", type=int, default=800)
    ap.add_argument("--min-valid", type=float, default=0.05)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-tiles", type=int, default=0, help="benchmark only")
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "1")))
    m = load_model(a.weights, a.min_size)
    name = Path(a.weights).stem
    out = Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(f"{a.slrm}/*.npz")):
        reg = Path(f).stem
        if a.only and reg not in a.only:
            continue
        dst = out / f"{reg}.json"
        if dst.exists() and not a.max_tiles:
            continue
        z = np.load(f)
        t0 = time.time()
        if a.max_tiles:
            zz = {"slrm_u8": z["slrm_u8"][:a.tile * 2, :a.tile * 2], "valid": z["valid"][:a.tile * 2, :a.tile * 2]}
            b, s, n = run_region(m, zz, a.tile, a.stride, a.min_valid)
            print(json.dumps({"bench_tiles": n, "s_per_tile": round((time.time() - t0) / max(n, 1), 2)}))
            return
        b, s, n = run_region(m, z, a.tile, a.stride, a.min_valid)
        left, top, res = float(z["left"]), float(z["top"]), float(z["res"])
        dets = [{"x": round(left + (bb[0] + bb[2]) / 2 * res, 2), "y": round(top - (bb[1] + bb[3]) / 2 * res, 2),
                 "w": round((bb[2] - bb[0]) * res, 2), "h": round((bb[3] - bb[1]) * res, 2),
                 "score": round(float(ss), 4)} for bb, ss in zip(b.astype(np.float64).tolist(), s.tolist())]
        json.dump({"region": reg, "crs": json.loads(str(z["crs"])), "tiles": n, "dets": dets},
                  open(str(dst) + ".tmp", "w"), ensure_ascii=False)
        Path(str(dst) + ".tmp").rename(dst)
        print(json.dumps({"region": reg, "tiles": n, "dets": len(dets), "ge05": int((s >= 0.5).sum()),
                          "s": round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
