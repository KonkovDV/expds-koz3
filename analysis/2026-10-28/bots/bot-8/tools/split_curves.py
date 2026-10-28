import sys, time, json, gc
sys.path.insert(0, sys.argv[3] if len(sys.argv)>3 else "/workspace/koz3_shared/bot-8/repo")
from pathlib import Path
import expds.metric as M
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-8/shm_cache/split")
mp = "/workspace/koz3_shared/bot-8/repo/materials/compute_metrics.py"
t=time.perf_counter(); gc.disable()
pred=json.loads(Path(sys.argv[1]).read_bytes()); gt=json.loads(Path(sys.argv[2]).read_bytes()); gc.enable()
print("json", time.perf_counter()-t)
try:
    import orjson
    t=time.perf_counter(); p2=orjson.loads(Path(sys.argv[1]).read_bytes()); print("orjson", time.perf_counter()-t, p2==pred); del p2
except ImportError: print("no orjson")
mod=M.load_official(mp); calc=mod.F1Calculator()
for rep in range(2):
    t=time.perf_counter(); P=M._collect(mod,calc,pred,False); t1=time.perf_counter(); G=M._collect(mod,calc,gt,True); t2=time.perf_counter()
    M.build_curves(pred,gt,mp); t3=time.perf_counter()
    print(f"rep{rep} collect_pred {t1-t:.3f} collect_gt {t2-t1:.3f} curves_cached {t3-t2:.3f}")
