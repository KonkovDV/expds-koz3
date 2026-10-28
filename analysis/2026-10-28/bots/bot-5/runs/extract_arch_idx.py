"""Индексы строк архитектуры в stack_arch.geojson (= строки table.pkl реранкера) и все участки стека."""
import json, sys, io, contextlib, ijson, numpy as np
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials")
with contextlib.redirect_stdout(io.StringIO()):
    import compute_metrics as cm
idx, regs = [], set()
TG = set(cm.F1Calculator().target_classes)
with open("/workspace/rerank/stack_arch.geojson", "rb") as fh:
    for n, f in enumerate(ijson.items(fh, "features.item", use_float=True)):
        p = f.get("properties") or {}
        c = cm.normalize_class_name(p.get("class_name", "unknown"))
        if c in TG and f.get("geometry"):
            regs.add(str(p.get("region_name", "default_source")))
        if c == "arkhitektury": idx.append(n)
np.save("arch_rows.npy", np.array(idx, np.int64))
json.dump({"n": n + 1, "pred_regions": sorted(regs)}, open("stack_regions.json", "w"), ensure_ascii=False)
print(n + 1, len(idx), len(regs))
