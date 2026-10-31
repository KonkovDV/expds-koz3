"""W5 shim: same API as analysis/2026-10-28/rerank/common.py, paths from env.
W5_ROOT = repo worktree (default: 3 levels up), W5_DATA = dir with stack_arch.geojson + gt_train_v2.geojson (+ work/)."""
import json, sys, time, gc, pickle, os
from pathlib import Path
ROOT = Path(os.environ.get('W5_ROOT', Path(__file__).resolve().parents[3])); sys.path.insert(0, str(ROOT))
METRIC = str(ROOT / 'materials' / 'compute_metrics.py')
from expds.metric import build_curves, sweep_groups, family_folds, honest_counts, score_counts, paired_bootstrap_counts, score_group_thresholds
from expds.naming import site_family
W = Path(os.environ.get('W5_DATA', '/workspace/koz3/w5data'))
STACK = os.environ.get('W5_STACK', 'stack_arch.geojson')
(W / 'work').mkdir(parents=True, exist_ok=True)

def load(p):
    gc.disable(); d = json.loads(Path(p).read_bytes()); gc.enable(); return d

def sets(gt):
    boxes = json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8'))
    regions = {(f.get('properties') or {}).get('region_name') for f in gt['features']}
    optical = {r for r in regions if r} - set(boxes)
    new = {l.strip() for l in (ROOT / 'notes' / 'regions_las_new.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
    return new, optical

def honest(curves, new, optical, k=10):
    regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
    folds = family_folds(regions, site_family, k)
    counts, info = honest_counts(curves, folds)
    s = [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]]
    return s, counts, info, folds

def insample(curves):
    res = sweep_groups(curves)
    taus = {}
    for cls, v in res.items():
        for g, gv in v['groups'].items(): taus.setdefault(g, {})[cls] = gv['threshold']
    return score_group_thresholds(curves, taus)[0], res
