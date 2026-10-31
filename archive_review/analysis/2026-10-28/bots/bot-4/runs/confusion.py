"""Путаница городища ↔ фортификации на стеке OOF (rerank/stack_arch.geojson.gz) при порогах базы."""
import json, collections
import numpy as np
from shapely.geometry import shape
from shapely.strtree import STRtree
from common import *
from expds.metric import build_curves, sweep_groups, apply_group_thresholds, score_group_thresholds, threshold_group

gt = gt_sub(load_gt()); pred = load_stack()
base = json.load(open("res_base.json"))
taus = {}
for c, v in base["all"].items():
    for g, gv in v["groups"].items():
        taus.setdefault(g, {})[c] = gv["threshold"]
kept = apply_group_thresholds(pred, taus, METRIC)
print("kept", collections.Counter((f["properties"]["class_name"], threshold_group(f["properties"])) for f in kept["features"]))

def hits_of(fc, gtfc):
    cv = build_curves(fc, gtfc, METRIC)
    return cv

# 1) свои попадания (на kept), по признаку fid
def tagged(fc):
    for i, f in enumerate(fc["features"]):
        f["properties"]["_i"] = i
    return fc
kept = tagged(kept)
own = build_curves(kept, gt, METRIC)
# соответствие: build_curves теряет индекс; восстановим попадания через порядок confidence внутри класса
def per_feat_hits(fc, gtfc, cls):
    feats = [f for f in fc["features"] if f["properties"]["class_name"] == cls]
    out = np.zeros(len(feats), bool)
    # жадно по одному объекту — дорого; вместо этого: по одному объекту против эталона класса без учёта занятости
    from expds.metric import load_official
    mod = load_official(METRIC); calc = mod.F1Calculator()
    g = [x[0] for x in calc._collect_by_class(gtfc, preserve_points=True).get(cls, [])]
    p = calc._collect_by_class({"type":"FeatureCollection","features":feats})
    pg = [x[0] for x in p.get(cls, [])]
    tree = STRtree(g)
    for k, geom in enumerate(pg):
        for j in tree.query(geom):
            ok, s = calc._match_quality(geom, g[int(j)])
            if ok:
                out[k] = True; break
    return feats, out

def relabel(feats, cls):
    return {"type":"FeatureCollection","features":[dict(f, properties=dict(f["properties"], class_name=cls)) for f in feats]}

res = {}
for a, b in (("fortifikatsii","gorodishcha"), ("gorodishcha","fortifikatsii")):
    fa, hit_own = per_feat_hits(kept, gt, a)              # совпадает ли с эталоном своего класса
    _, hit_other = per_feat_hits(relabel(fa, b), gt, b)   # совпал бы с эталоном другого класса
    grp = collections.Counter(); grp_conf = collections.Counter()
    for f, ho, hx in zip(fa, hit_own, hit_other):
        g = threshold_group(f["properties"])
        grp[(g, "own")] += int(ho); grp[(g, "other")] += int(hx); grp[(g, "other_not_own")] += int(hx and not ho); grp[(g,"n")] += 1
    res[f"{a}->{b}"] = {"kept": len(fa), "match_own": int(hit_own.sum()), "match_other": int(hit_other.sum()),
                        "match_other_not_own": int((hit_other & ~hit_own).sum()), "by_group": {f"{k[0]}:{k[1]}": v for k, v in sorted(grp.items())}}
    print(a, "->", b, res[f"{a}->{b}"])
    # оракул: перекинуть ложные-свои-но-верные-чужие в другой класс
    res[f"{a}->{b}"]["_swap_idx"] = [int(f["properties"]["_i"]) for f, ho, hx in zip(fa, hit_own, hit_other) if hx and not ho]
    # и просто убрать их (лучший детерминированный эффект — только FP своего класса)

# 2) Оракул «правильный класс»: каждой перепутанной выдаче — класс эталона
swap = {}
for k, v in res.items():
    tgt = k.split("->")[1]
    for i in v.pop("_swap_idx"):
        swap[i] = tgt
fixed = {"type":"FeatureCollection","features":[dict(f, properties=dict(f["properties"], class_name=swap.get(f["properties"]["_i"], f["properties"]["class_name"]))) for f in kept["features"]]}
removed = {"type":"FeatureCollection","features":[f for f in kept["features"] if f["properties"]["_i"] not in swap]}
for name, fc in (("kept_base", kept), ("oracle_swap", fixed), ("oracle_drop", removed)):
    cv = build_curves(fc, gt, METRIC)
    per = {}
    for c in CL:
        tp = int(cv[c].hit.sum()); n = len(cv[c].hit)
        from expds.metric import _f1
        per[c] = {"tp": tp, "fp": n - tp, "n_gt": cv[c].n_gt, "f1": float(_f1(tp, n - tp, cv[c].n_gt - tp))}
    contrib = sum(W[c]*per[c]["f1"] for c in CL)/100
    res[name] = {"per": per, "contrib": contrib}
    print(name, per, f"вклад {contrib:.5f}")

# 3) Пересечение эталонов
gfe = [f for f in gt["features"]]
from expds.metric import load_official
mod = load_official(METRIC)
G = collections.defaultdict(list)
for f in gfe:
    p = f["properties"]; c = mod.normalize_class_name(p.get("class_name"))
    gm = shape(f["geometry"])
    if gm.geom_type not in ("Polygon","MultiPolygon") or gm.is_empty: continue
    G[c].append((p.get("region_name"), p.get("markup_type"), gm.buffer(0)))
gor = G["gorodishcha"]; fort = G["fortifikatsii"]
tree = STRtree([x[2] for x in gor])
inside = 0; touch = 0; pairs = collections.Counter(); fort_regs = collections.Counter(); share=[]
for r, m, gm in fort:
    best = 0.0
    for j in tree.query(gm):
        rg, mg, gg = gor[int(j)]
        if rg != r: continue
        inter = gm.intersection(gg).area / max(gm.area, 1e-9)
        best = max(best, inter)
        if inter > 0: pairs[(m, mg)] += 1
    if best > 0: touch += 1; fort_regs[r] += 1
    if best >= 0.5: inside += 1
gor_with_fort = 0
ftree = STRtree([x[2] for x in fort])
for r, m, gg in gor:
    if any(fort[int(j)][0] == r and fort[int(j)][2].intersects(gg) for j in ftree.query(gg)): gor_with_fort += 1
res["gt_overlap"] = {"n_gor_poly": len(gor), "n_fort_poly": len(fort), "fort_touching_gor": touch, "fort_inside_gor_50": inside,
                     "gor_with_fort_inside": gor_with_fort, "pairs_markup(fort,gor)": {f"{a},{b}": v for (a,b), v in pairs.items()},
                     "regions": dict(fort_regs)}
print(res["gt_overlap"])
json.dump(res, open("res_confusion.json","w"), ensure_ascii=False, indent=1)
