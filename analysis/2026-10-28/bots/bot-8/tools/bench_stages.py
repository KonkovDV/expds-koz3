"""Время стадий по кривым: эталон (_reference) против быстрых, медиана из N повторов, ×SCALE.
python bench_stages.py PRED GT SCALE REPS OUT"""
import json, sys, time, statistics as S, pickle
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, "/workspace/koz3_shared/bot-8/repo")
import expds.metric as M
from expds.naming import site_family
M._SHM_CACHE_DIR = Path("/workspace/koz3_shared/bot-8/shm_cache_bench")
METRIC = "/workspace/koz3_shared/bot-8/repo/materials/compute_metrics.py"
pred, gt, scale, reps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
pk = Path("/workspace/koz3_shared/bot-8/bench/curves_every20.pkl")
if pk.exists():
    curves = pickle.loads(pk.read_bytes())
else:
    curves = M.build_curves(json.loads(Path(pred).read_bytes()), json.loads(Path(gt).read_bytes()), METRIC)
    pk.write_bytes(pickle.dumps(curves))
def tile(c, k):
    conf = np.tile(c.conf, k); hit = np.tile(c.hit, k)
    o = np.argsort(-conf, kind="stable")
    reg = (c.pred_region * k); grp = (c.pred_group * k)
    return M.ClassCurve(conf[o], hit[o], [reg[i] for i in o], c.n_gt * k,
                        Counter({r: v * k for r, v in c.gt_regions.items()}), [grp[i] for i in o])
cv = {cls: tile(c, scale) for cls, c in curves.items()}
regions = {r for c in cv.values() for r in c.pred_region} | {r for c in cv.values() for r in c.gt_regions}
folds = M.family_folds(regions, site_family, 10)
cnt, _ = M.honest_counts_reference(cv, folds)
thr = {c: 0.3 for c in cv}
stages = {
  "sweep_groups": (lambda: M.sweep_groups_reference(cv), lambda: M.sweep_groups(cv)),
  "honest_counts": (lambda: M.honest_counts_reference(cv, folds), lambda: M.honest_counts(cv, folds)),
  "bootstrap_200": (lambda: M.bootstrap_reference(cv, 200), lambda: M.bootstrap(cv, 200)),
  "paired_bootstrap_1000": (lambda: M.paired_bootstrap_reference(cv, cv, thr, {}, 1000), lambda: M.paired_bootstrap(cv, cv, thr, {}, 1000)),
  "paired_bootstrap_counts_1000": (lambda: M.paired_bootstrap_counts_reference(cnt, cnt, site_family, 1000), lambda: M.paired_bootstrap_counts(cnt, cnt, site_family, 1000)),
}
res = {"scale": scale, "n_pred": int(sum(len(c.conf) for c in cv.values())), "reps": reps}
for k, (fr, ff) in stages.items():
    tr, tf = [], []
    for _ in range(reps):
        t = time.perf_counter(); a = fr(); tr.append(time.perf_counter() - t)
        t = time.perf_counter(); b = ff(); tf.append(time.perf_counter() - t)
    same = json.dumps(a, default=str) == json.dumps(b, default=str)
    res[k] = {"ref": tr, "fast": tf, "ref_med": S.median(tr), "fast_med": S.median(tf),
              "speedup": S.median(tr) / S.median(tf), "identical": same}
    print(k, round(S.median(tr), 4), round(S.median(tf), 4), round(S.median(tr) / S.median(tf), 1), same, flush=True)
Path(out).write_text(json.dumps(res, indent=1))
