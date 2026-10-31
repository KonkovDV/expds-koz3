"""Побайтное равенство build_curves (новый) == build_curves_reference + _collect_reference, обе области.
python tools/eq_check.py PRED GT [--out OUT.json]"""
import sys, json, time, pickle, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("pred"); ap.add_argument("gt"); ap.add_argument("--out"); ap.add_argument("--cache", default="/workspace/koz3_shared/bot-8/geom_cache_eq")
a = ap.parse_args()
sys.path.insert(0, "/workspace/koz3_shared/bot-8/repo")
import os; os.environ["EXPDS_GEOM_CACHE"] = a.cache
import expds.metric as M
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-8/shm_cache_eq")
METRIC = "/workspace/koz3_shared/bot-8/repo/materials/compute_metrics.py"
pred = json.loads(Path(a.pred).read_bytes()); gt = json.loads(Path(a.gt).read_bytes())
def blob(c):
    return {k: (v.conf.tobytes(), v.hit.tobytes(), v.pred_region, v.n_gt, dict(v.gt_regions), v.pred_group) for k, v in c.items()}
res = {}
for scope in ("global", "site"):
    t = time.perf_counter(); new1 = M.build_curves(pred, gt, METRIC, scope=scope); t1 = time.perf_counter() - t
    t = time.perf_counter(); new2 = M.build_curves(pred, gt, METRIC, scope=scope); t2 = time.perf_counter() - t
    M._UTM_GEOM_CACHE.clear()
    fast_collect = M._collect; M._collect = M._collect_reference
    try:
        os.environ["EXPDS_GEOM_CACHE"] = "off"
        t = time.perf_counter(); ref = M.build_curves_reference(pred, gt, METRIC, scope=scope); t3 = time.perf_counter() - t
    finally:
        M._collect = fast_collect; os.environ["EXPDS_GEOM_CACHE"] = a.cache; M._UTM_GEOM_CACHE.clear()
    eq = pickle.dumps(blob(new1)) == pickle.dumps(blob(ref)) and pickle.dumps(blob(new2)) == pickle.dumps(blob(ref))
    hits = {k: int(v.hit.sum()) for k, v in ref.items()}
    res[scope] = {"equal": eq, "t_new_first": round(t1, 3), "t_new_repeat": round(t2, 3), "t_reference": round(t3, 3), "hits": hits,
                  "n_pred": {k: len(v.conf) for k, v in ref.items()}}
    print(scope, res[scope], flush=True)
if a.out: Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
