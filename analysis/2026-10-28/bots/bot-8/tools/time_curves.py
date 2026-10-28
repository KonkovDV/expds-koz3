import json, sys, time, gc, os, statistics
from pathlib import Path
sys.path.insert(0, os.environ.get("REPO", "/workspace/koz3_shared/bot-8/repo"))
import expds.metric as M
pred = json.loads(Path(sys.argv[1]).read_bytes()); gt = json.loads(Path(sys.argv[2]).read_bytes())
METRIC = "/workspace/koz3_shared/bot-8/repo/materials/compute_metrics.py"
M.load_official(METRIC)
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-8/shm_cache_off")  # не существует -> только запись; чистим
import gc
if os.environ.get("NOGC"): gc.disable()
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ts = []
for r in range(reps):
    M._UTM_GEOM_CACHE.clear()
    import shutil; shutil.rmtree(M._SHM_CACHE_DIR, ignore_errors=True)
    if os.environ.get("NOSHM"): M._SHM_CACHE_DIR = Path("/nonexistent_dir_x/y")
    t = time.perf_counter(); c = M.build_curves(pred, gt, METRIC); ts.append(time.perf_counter() - t)
print(json.dumps({"times": ts, "median": statistics.median(ts), "hits": {k: int(v.hit.sum()) for k, v in c.items()}}))
