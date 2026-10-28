"""Прогон шагов scoreboard.py --honest на реальной выборке стека (без build_stack: стек уже собран).
python harness.py PRED GT [--boot N] [--honest K] [--stages a,b,...]"""
import argparse, gc, json, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("gt")
ap.add_argument("--repo", default="/workspace/koz3_shared/bot-8/repo")
ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--honest", type=int, default=10)
ap.add_argument("--out", default=None)
a = ap.parse_args()
sys.path.insert(0, a.repo)
import expds.metric as M
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-8/shm_cache")
from expds.naming import site_family
METRIC = str(Path(a.repo) / "materials" / "compute_metrics.py")
T = {}
def tic(k, t0): T[k] = T.get(k, 0) + time.perf_counter() - t0
gc.disable(); pred = json.loads(Path(a.pred).read_bytes()); gt = json.loads(Path(a.gt).read_bytes()); gc.enable()
t = time.perf_counter(); curves = M.build_curves(pred, gt, METRIC); tic("build_curves", t)
t = time.perf_counter(); res = M.sweep_groups(curves); tic("sweep_groups", t)
taus = {}
for cls, v in res.items():
    for g, gv in v["groups"].items(): taus.setdefault(g, {})[cls] = gv["threshold"]
t = time.perf_counter(); kept = M.apply_group_thresholds(pred, taus, METRIC); tic("apply_thr", t)
t = time.perf_counter(); ck = M.build_curves(kept, gt, METRIC); s_all = M.score_at(ck, {})[0]; tic("final_score", t)
regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
folds = M.family_folds(regions, site_family, a.honest)
t = time.perf_counter(); counts, info = M.honest_counts(curves, folds); tic("honest_counts", t)
h = M.score_counts(counts)[0]
t = time.perf_counter(); pb = M.paired_bootstrap(ck, curves, {}, {}, a.boot); tic("paired_bootstrap", t)
counts_b, _ = M.honest_counts(ck, folds)
t = time.perf_counter(); pbc = M.paired_bootstrap_counts(counts, counts_b, site_family, a.boot); tic("paired_bootstrap_counts", t)
out = {"n_pred": len(pred["features"]), "n_gt": len(gt["features"]), "folds": len(folds), "in_sample": s_all,
       "honest": h, "info": info, "pb": pb["score"], "pbc": pbc, "times": T,
       "hits": {c: int(v.hit.sum()) for c, v in curves.items()}}
print(json.dumps(out, ensure_ascii=False, indent=1))
if a.out: Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
