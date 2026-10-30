#!/usr/bin/env python3
"""exec A: apply honest per-fold thresholds (taus from sweep_groups on the other geo5 folds, same as
pix_clean_eval/honest_counts, unseen=drop) to a geo geojson -> thresholded geojson for raw P/R gate."""
import sys, json
sys.path.insert(0, "/workspace/koz3/wt/w4-pix"); sys.path.insert(0, "/workspace/koz3/wt/w4-pix/scripts")
from expds import metric
import pix_clean_eval as E
gt_p, src, out = sys.argv[1:4]
regions = {f["properties"]["region_name"] for f in json.load(open(gt_p))["features"]
           if f["properties"].get("region_name", "").split("_")[0] in {str(i) for i in range(200, 218)}}
gt = E._load(gt_p, regions); pred = E._load(src, regions)
folds = E.folds_of(regions); curves = metric.build_curves(pred, gt)
mod = metric.load_official(None); every = set().union(*folds.values()); keep = []
for k, held in folds.items():
    if not held: continue
    res = metric.sweep_groups({c: metric.curve_where(cv, every - held) for c, cv in curves.items()})
    taus = {}
    for cls, v in res.items():
        for g, gv in v["groups"].items(): taus.setdefault(g, {})[cls] = gv["threshold"]
    n0 = 0
    for f in pred["features"]:
        p = f["properties"]
        if p["region_name"] not in held: continue
        n0 += 1
        t = taus.get(metric.threshold_group(p), {}).get(mod.normalize_class_name(p["class_name"]))
        if t is not None and mod._get_confidence(f) >= t: keep.append(f)
    print(k, "taus", {g: {c: round(x, 4) for c, x in d.items()} for g, d in taus.items()}, "in", n0)
for i, f in enumerate(keep): f["properties"]["fid"] = i
pred["features"] = keep; json.dump(pred, open(out, "w"), ensure_ascii=False); print("kept", len(keep), out)
