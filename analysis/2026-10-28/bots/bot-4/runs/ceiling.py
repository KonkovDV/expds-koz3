"""Потолок арбитра и грубого прохода: сколько эталонных городищ, не найденных базой при её порогах,
вообще достижимы кругами fort2gor / gor_coarse (без учёта точности)."""
import json, collections
import numpy as np
from shapely.strtree import STRtree
from common import *
from expds import gorfort as gf
from expds.metric import load_official, apply_group_thresholds, _f1
gt = gt_sub(load_gt()); pred = load_stack()
mod = load_official(METRIC); calc = mod.F1Calculator()
G = [x[0] for x in calc._collect_by_class(gt, preserve_points=True)["gorodishcha"]]
GREG = [f["properties"].get("region_name") for f in gt["features"] if mod.normalize_class_name(f["properties"].get("class_name")) == "gorodishcha"]
tree = STRtree(G)
def hits(feats):
    geoms = [x[0] for x in calc._collect_by_class({"type": "FeatureCollection", "features": feats}).get("gorodishcha", [])]
    out = []
    for g in geoms:
        out.append({int(j) for j in tree.query(g) if calc._match_quality(g, G[int(j)])[0]})
    return out
base = json.load(open("res_base.json"))
taus = {}
for c, v in base["all"].items():
    for g, gv in v["groups"].items():
        taus.setdefault(g, {})[c] = gv["threshold"]
kept = apply_group_thresholds(pred, taus, METRIC)
kg = [f for f in kept["features"] if f["properties"]["class_name"] == "gorodishcha"]
A = set().union(*hits(kg))
p = gf.GorFortParams(); t = gf.table(pred, p)
q, F = gf.fort_features(t, p)
hf = hits(gf.circles(t["x"][q], t["y"][q], t["k"][q], np.ones(q.sum()), t["region"][q], "fort2gor"))
B = set().union(*hf)
cl = gf.clusters(t, p); cx, cy, _ = gf.cluster_features(t, cl)
ck = np.array([float(np.median(t["k"][m])) for m in cl]); creg = np.array([t["region"][m[0]] for m in cl], dtype=object)
hc = hits(gf.circles(cx, cy, ck, np.ones(len(cl)), creg, "gor_coarse"))
C = set().union(*hc)
allg = [f for f in pred["features"] if f["properties"]["class_name"] == "gorodishcha"]
P = set().union(*hits(allg))
grp = t["group"][q]
newB = B - A; newC = C - A
res = {"n_gt": len(G), "hit_kept_base_any": len(A), "hit_gor_pool_any": len(P),
       "fort2gor_any": len(B), "fort2gor_new_vs_kept": len(newB), "fort2gor_new_vs_gor_pool": len(B - P),
       "fort2gor_new_by_group": dict(collections.Counter(g for h, g in zip(hf, grp) for j in h if j in newB)),
       "coarse_any": len(C), "coarse_new_vs_kept": len(newC), "coarse_new_vs_gor_pool": len(C - P),
       "new_union": len(newB | newC), "new_union_regions": dict(collections.Counter(GREG[j] for j in newB | newC))}
# лучший случай: база (tp 31, fp 70 на kept) + ровно по одному верному кругу на новый эталон, без лишних
tp0, fp0 = 31, 70
for k, n in (("fort2gor", len(newB)), ("coarse", len(newC)), ("union", len(newB | newC))):
    f1 = _f1(tp0 + n, fp0, len(G) - tp0 - n)
    res[f"oracle_{k}_gor_f1"] = float(f1); res[f"oracle_{k}_delta_total"] = float(15 * (f1 - 0.33696) / 100)
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open("res_ceiling.json", "w"), ensure_ascii=False, indent=1)
