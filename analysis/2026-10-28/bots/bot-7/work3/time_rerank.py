"""Время и память rerank_arch.apply на самой большой семье стека (bot-10/work/fam: Курганы — больше всего архитектуры, 12 379; Вышегжа — больше всего объектов)."""
import resource
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))
from shapely.geometry import shape
from expds import rerank_arch
feats = []
for line in (l for fn in ("Курганы", "Вышегжа") for l in open(f"/workspace/koz3_shared/bot-10/work/fam/{fn}.jsonl", encoding="utf-8")):
    d = json.loads(line); p = d["properties"]
    f = {k: v for k, v in p.items() if k not in ("fid", "crs", "original_crs")}
    f["geom"] = shape(d["geometry"]); feats.append(f)
regs = {}
for f in feats: regs.setdefault(f["region_name"], []).append(f)
print("rows", len(feats), "regions", len(regs), "arch", sum(f["class_name"] == "arkhitektury" for f in feats), flush=True)
opts = {"enabled": True, "model": "models/rerank_arch_lgbm.txt", "meta": "models/rerank_arch_lgbm.json"}
for name, fs in [("all", feats)] + sorted(regs.items(), key=lambda kv: -len(kv[1]))[:3]:
    t = time.monotonic(); rest, kept = rerank_arch.apply(fs, opts)
    print(name, len(fs), "rest", len(rest), "kept", len(kept), f"{time.monotonic() - t:.1f} s", flush=True)
print("maxrss_mb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
