#!/usr/bin/env python3
"""exec A (gnz): fit pix_clean finds from an existing feat cache with site-family weights + raster excludes (clean; same LGB params as retrain.py)."""
import sys, json, argparse, hashlib, numpy as np
from pathlib import Path
sys.path.insert(0, "/workspace/koz3/wt/w4-pix"); sys.path.insert(0, "/workspace/koz3/wt/w4-pix/scripts")
import pix_clean_train as M
from expds import pix_clean as pc
import lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument("--cache"); ap.add_argument("--out"); ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--gnz-w", type=float, default=3.0); ap.add_argument("--drop-raster-re", default="")
ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
import re
cache = Path(a.cache); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
idx = json.loads((cache / "index_s0.json").read_text())
Xs, ys, ws, regs = [], [], [], set()
for info in idx:
    if info["region"].split("_")[0] in a.exclude or "finds_points" not in info["classes"] or info["mt"] != "maggr": continue
    if a.drop_raster_re and re.search(a.drop_raster_re, info["raster"]): print("drop", info["raster"]); continue
    z = np.load(cache / info["file"]); X, y = z["X_finds_points"], z["y_finds_points"]
    w = a.gnz_w if "Гнёздово" in info["region"] else 1.0
    Xs.append(X); ys.append(y); ws.append(np.full(len(y), w, np.float32)); regs.add(info["region"])
X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
P = dict(M.LGB_PARAMS, num_leaves=63, min_data_in_leaf=100, seed=0, num_threads=a.threads)
b = lgb.train(P, lgb.Dataset(X, label=y, weight=w, feature_name=pc.feature_names()), num_boost_round=a.rounds)
b.save_model(str(out / "finds.txt"))
meta = {"feature_version": pc.FEATURE_VERSION, "feature_names": pc.feature_names(), "seed": 0, "sample_seed": 0,
        "exclude_regions": sorted(a.exclude), "decode": dict(pc.DEFAULT_DECODE), "params": dict(P, num_rounds=a.rounds),
        "classes": {"finds_points": {"rows": int(len(y)), "pos": int(y.sum()), "regions": sorted(regs), "gnz_w": a.gnz_w,
                    "drop_raster_re": a.drop_raster_re, "sha256": hashlib.sha256((out / "finds.txt").read_bytes()).hexdigest()}},
        "provenance": "clean room (exec A gnz): fit_w.py over feat cache, no EN code"}
(out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
print("ok", out, len(y), int(y.sum()))
