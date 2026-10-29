"""bot-6 v4: смоук dfine_or bot-9 (f1) на 185_Изборск Or, CPU 2 потока. Печатает n_det по классам, RSS, с/окно."""
import json, resource, sys, time
from collections import Counter
from pathlib import Path
REPO = Path("/workspace/koz3_shared/bot-6/repo"); ROOT = Path(sys.argv[1]); TIF = Path(sys.argv[2])
sys.path.insert(0, str(REPO))
import torch
torch.set_num_threads(2)
from expds import dfine_aeor as A, sp_dfine as S
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
cpu = lambda: time.process_time()
cfg = {"dfine_aeor": {"enabled": True, "markups": ["or"], "weights": {"or": ["models/dfine_or"]},
                      "thresholds_file": "models/thresholds_stub_v4.json"},
       "thresholds_by_markup": {"dfine_or": {"gorodishcha": 1.01}}}   # как в configs/default.toml
import os
st = A.settings(cfg); st["device"] = "cpu"
if os.environ.get("B6_BATCH"): st["batch"] = dict(st["batch"], **{"or": int(os.environ["B6_BATCH"])})
print("batch or =", st["batch"]["or"], flush=True); dev = torch.device("cpu")
used = A.load_thresholds_file(cfg, ROOT); ok, miss = A.thresholds_ready(cfg, "or")
thr = cfg["thresholds_by_markup"]["dfine_or"]
print("thresholds_file used:", used, "| итог dfine_or:", thr, "| ready:", ok, miss, flush=True)
t0, c0 = time.monotonic(), cpu()
_, _, info = S.import_dfine(ROOT)
print("import_dfine:", {k: info.get(k) for k in ("pin", "vendor_present", "source", "version")}, flush=True)
models, labels = A._load(st, "or", ROOT, {"device": dev})
print(f"LOAD wall={time.monotonic()-t0:.1f}s cpu={cpu()-c0:.1f}s rss_mb={rss()} labels={labels}", flush=True)
res = {}
for level in sys.argv[3:]:
    t0, c0 = time.monotonic(), cpu()
    B, L, C, T0, done, total = A.detect_image(models, TIF, S.utm_for(TIF, None), st, "or", level, dev)
    wall, c = time.monotonic() - t0, cpu() - c0
    cls = [labels[int(k)] if not isinstance(k, str) else k for k in C]
    raw = Counter(cls); kept = Counter(k for k, s in zip(cls, L) if float(s) >= thr.get(k, 9))
    mx = {k: round(max(float(s) for kk, s in zip(cls, L) if kk == k), 4) for k in raw}
    r = {"n_det": len(L), "by_class_min_score": dict(raw), "max_by_class": mx, "kept_stub": dict(kept),
         "windows": [int(done), int(total)], "wall_s": round(wall, 1), "cpu_s": round(c, 1),
         "wall_s_per_window": round(wall / max(done, 1), 1), "rss_mb": rss()}
    res[level] = r
    print(f"LEVEL {level}: " + json.dumps(r, ensure_ascii=False), flush=True)
print("SMOKE_JSON=" + json.dumps(res, ensure_ascii=False))
