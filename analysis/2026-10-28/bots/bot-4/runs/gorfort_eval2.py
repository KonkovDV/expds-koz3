"""Оценка арбитра fort→gor и грубого прохода gor_coarse (expds/gorfort.py) на стеке OOF.

Честно: классификатор обучается на 9 фолдах семей участков и предсказывает 10-й;
пороги новой группы подбирает honest_counts на тех же фолдах (common.evaluate).
"""
import json, sys, time, collections
import numpy as np
from shapely.strtree import STRtree
from shapely.geometry import shape
from common import *
from expds import gorfort as gf
from expds.metric import load_official, family_folds
from expds.naming import site_family

t0 = time.time()
gt = gt_sub(load_gt()); pred = load_stack()
mod = load_official(METRIC); calc = mod.F1Calculator()
G = [x[0] for x in calc._collect_by_class(gt, preserve_points=True)["gorodishcha"]]
GT_TREE = STRtree(G)
p = gf.GorFortParams()
t = gf.table(pred, p)
print("table", len(t["cls"]), collections.Counter(zip(t["cls"], t["group"])), f"{time.time()-t0:.0f}s", flush=True)

def labels(feats):
    fc = {"type": "FeatureCollection", "features": feats}
    geoms = [x[0] for x in calc._collect_by_class(fc).get("gorodishcha", [])]
    assert len(geoms) == len(feats)
    y = np.zeros(len(feats), bool)
    for i, g in enumerate(geoms):
        for j in GT_TREE.query(g):
            if calc._match_quality(g, G[int(j)])[0]:
                y[i] = True; break
    return y

regions = {f["properties"]["region_name"] for f in pred["features"]} | {f["properties"].get("region_name") for f in gt["features"]}
folds = family_folds(regions, site_family, 10)
fold_of = {r: i for i, fs in enumerate(folds) for r in fs}

def oof(X, y, reg, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    f = np.array([fold_of.get(r, -1) for r in reg])
    out = np.zeros(len(y))
    for k in range(len(folds)):
        te = f == k; tr = ~te
        if not te.any():
            continue
        if y[tr].sum() < 3:
            continue
        if kind == "lr":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))
        else:
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=20, l2_regularization=1.0)
        m.fit(X[tr], y[tr]); out[te] = m.predict_proba(X[te])[:, 1]
    return out

from sklearn.metrics import roc_auc_score, average_precision_score
res = {"folds": len(folds)}
# --- арбитр fort→gor
q, F = gf.fort_features(t, p)
names = sorted(F); X = np.c_[[F[n] for n in names]].T
xs, ys, ks, regs = t["x"][q], t["y"][q], t["k"][q], t["region"][q]
y = labels(gf.circles(xs, ys, ks, np.ones(q.sum()), regs, "fort2gor"))
print("fort2gor: n", int(q.sum()), "pos", int(y.sum()), f"{time.time()-t0:.0f}s", flush=True)
scores = {"rule": gf.rule_score_fort(F), "lr": oof(X, y, regs, "lr"), "gbm": oof(X, y, regs, "gbm"), "conf": F["conf"]}
res["fort2gor"] = {"n": int(q.sum()), "pos": int(y.sum()), "by_group_pos": {g: [int((t["group"][q] == g).sum()), int(y[t["group"][q] == g].sum())] for g in p.fort_groups}}
for k, s in scores.items():
    res["fort2gor"][f"auc_{k}"] = float(roc_auc_score(y, s)); res["fort2gor"][f"ap_{k}"] = float(average_precision_score(y, s))
print(json.dumps(res["fort2gor"]), flush=True)
add_f2g = {k: gf.circles(xs, ys, ks, s, regs, "fort2gor") for k, s in scores.items() if k != "conf"}
# --- грубый проход
cl = gf.clusters(t, p)
cx, cy, F2 = gf.cluster_features(t, cl)
n2 = sorted(F2); X2 = np.c_[[F2[n] for n in n2]].T
ck = np.array([float(np.median(t["k"][m])) for m in cl]); creg = np.array([t["region"][m[0]] for m in cl], dtype=object)
y2 = labels(gf.circles(cx, cy, ck, np.ones(len(cl)), creg, "gor_coarse"))
gor_hit_by_coarse = None
print("gor_coarse: clusters", len(cl), "pos", int(y2.sum()), f"{time.time()-t0:.0f}s", flush=True)
s2 = {"lr": oof(X2, y2, creg, "lr"), "gbm": oof(X2, y2, creg, "gbm")}
res["gor_coarse"] = {"n": len(cl), "pos": int(y2.sum())}
for k, s in s2.items():
    res["gor_coarse"][f"auc_{k}"] = float(roc_auc_score(y2, s)); res["gor_coarse"][f"ap_{k}"] = float(average_precision_score(y2, s))
print(json.dumps(res["gor_coarse"]), flush=True)
add_c = {k: gf.circles(cx, cy, ck, s, creg, "gor_coarse") for k, s in s2.items()}

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
import copy
def run(tag, extra):
    extra = copy.deepcopy(extra)
    for f in extra:
        f["properties"]["confidence"] = round(f["properties"]["confidence"] * SCALE, 6)
    fc = {"type": "FeatureCollection", "features": pred["features"] + extra}
    out, _, _ = evaluate(fc, gt, 10, "family", tag)
    return {"n_added": len(extra), "contrib_all": out["contrib_all"], "contrib_honest": out["contrib_honest"],
            "all": {c: {"f1": out["all"][c]["f1"], "groups": out["all"][c]["groups"]} for c in out["all"]}, "honest": out["honest"]}
res["scale"] = SCALE
res["eval"] = {"base": run("base", [])}
grp = t["group"][q]
for kind in ("gbm", "lr", "rule"):
    s = scores[kind]
    for name, sel in (("all", np.ones(len(s), bool)), ("li_only", grp != "ae_fort")):
        keep = sel & gf.nms(xs, ys, ks, np.where(sel, s, -1), regs, 60.0)
        res["eval"][f"f2g_{kind}_{name}_nms"] = run(f"f2g_{kind}_{name}_nms", gf.circles(xs[keep], ys[keep], ks[keep], s[keep], regs[keep], "fort2gor"))
for kind in ("gbm", "lr"):
    s = s2[kind]; keep = gf.nms(cx, cy, ck, s, creg, 60.0)
    res["eval"][f"coarse_{kind}_nms"] = run(f"coarse_{kind}_nms", gf.circles(cx[keep], cy[keep], ck[keep], s[keep], creg[keep], "gor_coarse"))
for k, v in res["eval"].items():
    print(f"{k:24s} n {v['n_added']:6d} all {v['contrib_all']:.5f} ({v['contrib_all']-res['eval']['base']['contrib_all']:+.5f})  honest {v['contrib_honest']:.5f} ({v['contrib_honest']-res['eval']['base']['contrib_honest']:+.5f})  gorH {v['honest']['gorodishcha']['f1']:.4f}")
json.dump(res, open(f"res_gorfort_nms_s{SCALE:g}.json", "w"), ensure_ascii=False, indent=1, default=float)
print(f"done {time.time()-t0:.0f}s")
