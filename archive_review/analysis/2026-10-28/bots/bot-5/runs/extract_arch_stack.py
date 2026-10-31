"""Поток stack_arch.geojson (1,1 ГБ, /workspace/rerank) → только архитектура; gt_train_v2 → только архитектура."""
import json, sys, io, contextlib, collections, ijson
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials")
with contextlib.redirect_stdout(io.StringIO()):
    import compute_metrics as cm
A = "arkhitektury"
out, groups = [], collections.Counter()
with open("/workspace/rerank/stack_arch.geojson", "rb") as fh:
    n = 0
    for f in ijson.items(fh, "features.item", use_float=True):
        n += 1
        if cm.normalize_class_name((f.get("properties") or {}).get("class_name", "")) == A:
            out.append(f); p = f["properties"]; groups[p.get("threshold_group") or p.get("markup_type")] += 1
json.dump({"type": "FeatureCollection", "features": out}, open("stack_arch_only.geojson", "w"), ensure_ascii=False)
gt = json.load(open("/workspace/rerank/gt_train_v2.geojson"))
ga = [f for f in gt["features"] if cm.normalize_class_name((f.get("properties") or {}).get("class_name", "")) == A]
json.dump({"type": "FeatureCollection", "features": ga}, open("gt_train_v2_arch.geojson", "w"), ensure_ascii=False)
print("всего в стеке", n, "архитектуры", len(out), dict(groups), "эталон арх.", len(ga))
