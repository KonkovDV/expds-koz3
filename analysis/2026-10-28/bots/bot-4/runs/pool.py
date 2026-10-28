"""Потолок пула: какие эталоны городищ/фортификаций ловит хоть один кандидат своего класса (любой уверенности) и какие — только кандидаты другого класса."""
import json, collections
from shapely.strtree import STRtree
from common import *
from expds.metric import load_official, threshold_group
mod = load_official(METRIC); calc = mod.F1Calculator()
gt = gt_sub(load_gt()); pred = load_stack()
G = calc._collect_by_class(gt, preserve_points=True)
P = {}
for pcls in CL:
    feats = [f for f in pred["features"] if f["properties"]["class_name"] == pcls]
    geoms = [x[0] for x in calc._collect_by_class({"type": "FeatureCollection", "features": feats}).get(pcls, [])]
    assert len(geoms) == len(feats)
    P[pcls] = (feats, geoms)
out = {}
for gcls in CL:
    g = [x[0] for x in G[gcls]]
    tree = STRtree(g)
    hitby = collections.defaultdict(set)
    for pcls in CL:
        feats, geoms = P[pcls]
        for f, pg in zip(feats, geoms):
            for j in tree.query(pg):
                ok, s = calc._match_quality(pg, g[int(j)])
                if ok:
                    hitby[int(j)].add(pcls + ":" + threshold_group(f["properties"]))
    own = sum(1 for j in range(len(g)) if any(s.startswith(gcls) for s in hitby[j]))
    other_only = [j for j in range(len(g)) if hitby[j] and not any(s.startswith(gcls) for s in hitby[j])]
    out[gcls] = {"n_gt": len(g), "hit_by_own_class_pool": own, "hit_only_by_other_class_pool": len(other_only),
                 "other_only_groups": dict(collections.Counter(s for j in other_only for s in hitby[j])),
                 "hit_by_other_class_pool": sum(1 for j in range(len(g)) if any(not s.startswith(gcls) for s in hitby[j]))}
    print(gcls, out[gcls], flush=True)
json.dump(out, open("res_pool.json", "w"), ensure_ascii=False, indent=1)
