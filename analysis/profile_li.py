"""Время по этапам лидара на настоящих LAS: где тратится время при инференсе."""
import sys
import time

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")

import numpy as np

from expds import accel, candidates
from expds.catalog import read_utm_json, scan_root
from expds.infer_nn import Segmenter
from expds.lidar import fill_interior_holes, relief_channels
from expds.pipeline import _params, load_config

ROOT = r"F:\expds3\train-expds-3"
NAMES = sys.argv[1:] or ["021_Новенькое", "101_Гнёздово_Днепровская"]

cfg = load_config(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3\configs\default.toml")
cand_p = _params(candidates.CandidateParams, cfg.get("candidates"))
seg = Segmenter(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3\models\li_r18_hard2.pt")
regions = {r.name: r for r in scan_root(ROOT)}
print("backend", accel.backend(), flush=True)

for name in NAMES:
    reg = regions[name]
    crs = read_utm_json(reg.utm_json)
    for las in reg.las:
        t = {}
        s = time.perf_counter()
        dem, info = accel.las_grid(las, 0.1, 2, 60_000_000)
        t["las_grid"] = time.perf_counter() - s
        s = time.perf_counter()
        valid = np.isfinite(dem)
        hull = accel.convex_hull_mask(valid)
        t["hull"] = time.perf_counter() - s
        s = time.perf_counter()
        filled = fill_interior_holes(dem, method="delaunay")
        t["fill_total"] = time.perf_counter() - s
        from expds.lidar import Dem

        d = Dem(dem=filled, left=info["left"], top=info["top"], res=info["res"], crs=crs, info=info)
        s = time.perf_counter()
        img, tr = relief_channels(d, seg.res, tuple(seg.meta["li_tags"]))
        t["channels"] = time.perf_counter() - s
        s = time.perf_counter()
        probs = seg.predict(img)
        t["predict_gpu"] = time.perf_counter() - s
        s = time.perf_counter()
        c = candidates.detect(d, cand_p)
        t["detect"] = time.perf_counter() - s
        s = time.perf_counter()
        sp = candidates.sample_probs(c, probs[:2], tr)
        t["sample"] = time.perf_counter() - s
        s = time.perf_counter()
        scores = np.column_stack([np.zeros(len(sp), np.float32), sp])
        feats = candidates.to_features(c, scores, crs, name, cand_p, source=las.name)
        t["features"] = time.perf_counter() - s
        total = sum(t.values())
        shape = f"{dem.shape[1]}x{dem.shape[0]}@{info['res']:.2g}"
        parts = "  ".join(f"{k} {v:.2f}" for k, v in t.items())
        print(f"{name} {shape} peaks {len(c['x'])} feats {len(feats)} total {total:.2f}s | {parts}", flush=True)
        del dem, filled, img, probs, c, feats
