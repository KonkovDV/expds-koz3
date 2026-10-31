"""Честные k10 пороги (как M.honest_counts) → TP/FP по (группа, класс) для групп ae*/or* и итог."""
import sys, json, pickle, collections
from pathlib import Path
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo"); sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo/scripts")
import numpy as np
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
cv = load_curves(Path(sys.argv[1])); pol = sys.argv[2] if len(sys.argv) > 2 else "drop"
regions = regions_of(cv); folds = M.family_folds(regions, site_family, 10)
every = set().union(*folds); agg = collections.Counter(); taus_all = collections.defaultdict(list)
for held in folds:
    res = M.sweep_groups({c: M.curve_where(x, every - held) for c, x in cv.items()})
    taus = {}
    for cls, v in res.items():
        for g, gv in v["groups"].items(): taus.setdefault(g, {})[cls] = gv["threshold"]
    for cls, c in cv.items():
        hc = M.curve_where(c, held)
        for g, conf, hit in zip(hc.pred_group, hc.conf, hc.hit):
            t = taus.get(g, {}).get(cls)
            if t is None: t = np.inf if pol == "drop" else -np.inf
            if conf >= t: agg[(g, cls, "tp" if hit else "fp")] += 1
        for g in set(hc.pred_group):
            if g in taus and cls in taus[g]: taus_all[(g, cls)].append(round(float(taus[g][cls]), 4))
out = {}
for (g, cls, k), n in sorted(agg.items()):
    if g.startswith(("ae", "or", "li_ae")): out.setdefault(f"{g}|{cls}", {})[k] = n
for k in out: out[k]["tau_med"] = float(np.median(taus_all.get(tuple(k.split("|")), [np.nan])))
print(json.dumps(out, ensure_ascii=False, indent=0))
