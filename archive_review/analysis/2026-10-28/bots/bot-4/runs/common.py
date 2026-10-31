import gzip, json, sys, os
sys.path.insert(0, "/workspace/koz3_shared/bot-4/repo")
METRIC = "/workspace/koz3_shared/bot-4/repo/materials/compute_metrics.py"
os.environ["EXPDS_METRIC"] = METRIC
from pathlib import Path as _P
import expds.metric as _em
_em._SHM_CACHE_DIR = _P("/workspace/koz3_shared/bot-4/work/utm_cache")  # не писать в общий /dev/shm
CL = ("gorodishcha", "fortifikatsii")
W = {"gorodishcha": 15.0, "fortifikatsii": 5.0}

def load_gt():
    gt = json.load(gzip.open("/workspace/rerank/gt_train_v2.geojson.gz"))
    return gt

def gt_sub(gt, classes=CL):
    from expds.metric import load_official
    mod = load_official(METRIC)
    feats = [f for f in gt["features"] if mod.normalize_class_name((f.get("properties") or {}).get("class_name", "")) in classes]
    return {"type": "FeatureCollection", "features": feats}

def load_stack():
    return json.load(open("/workspace/koz3_shared/bot-4/work/stack_gor_fort.geojson"))

def evaluate(pred, gt, folds_k=10, by="family", tag=""):
    from expds.metric import build_curves, sweep_groups, family_folds, honest_counts, score_counts
    from expds.naming import site_family
    curves = build_curves(pred, gt, METRIC)
    curves = {c: v for c, v in curves.items() if c in CL}
    res = sweep_groups(curves)
    regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
    fam = site_family if by == "family" else (lambda r: r)
    folds = family_folds(regions, fam, folds_k)
    counts, info = honest_counts(curves, folds)
    _, per = score_counts(counts)
    out = {"tag": tag, "all": {c: res[c] for c in res}, "honest": per, "honest_info": info}
    contrib_all = sum(W[c] * res[c]["f1"] for c in res) / 100
    contrib_h = sum(W[c] * per[c]["f1"] for c in per) / 100
    out["contrib_all"] = contrib_all; out["contrib_honest"] = contrib_h
    print(f"[{tag}] ALL: " + "; ".join(f"{c} F1 {res[c]['f1']:.4f}" for c in res) + f" -> вклад {contrib_all:.5f}")
    print(f"[{tag}] HONEST({info['folds']} фолдов {by}): " + "; ".join(f"{c} F1 {per[c]['f1']:.4f} tp {per[c]['tp']:.0f} fp {per[c]['fp']:.0f}" for c in per) + f" -> вклад {contrib_h:.5f}")
    for c in res:
        for g, gv in res[c]["groups"].items():
            print(f"    {c[:5]} {g:10s} thr {gv['threshold']:.6f} kept {gv['kept']:5d} tp {gv['tp']}")
    return out, curves, counts
