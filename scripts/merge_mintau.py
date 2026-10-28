"""bot-2, merge-check: минимальный честный порог (группа, класс) по 10 фолдам A (оба режима сопоставления) —
фильтр для чувствительности S (объекты ниже ни в одном фолде не выдаются и склеиться не могут)."""
import json, pickle, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "/workspace/expds-koz3")
from expds.metric import ClassCurve, sweep_groups, curve_where, family_folds
from expds.naming import site_family
R = pickle.load(open("/workspace/koz3_shared/bot-2/work/fusion_prep.pkl", "rb"))
B = pickle.load(open("/workspace/rerank/work/base_honest.pkl", "rb"))
CL = R["classes"]; every = set().union(*B["folds"]); folds = family_folds(every, site_family, 10)
def greedy(c, mode):
    P, pr = R["P"][c], R["pairs"][c]; ceff = P["conf"]
    m = pr["same"] if mode == "region" else np.ones(len(pr["i"]), bool)
    pi, pj, ps = pr["i"][m], pr["j"][m], pr["s"][m]; hit = np.zeros(len(ceff), bool)
    if not len(pi): return hit
    o = np.lexsort((pj, pi)); pi, pj, ps = pi[o], pj[o], ps[o]
    st = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]; en = np.r_[st[1:], len(pi)]
    preds = pi[st]; order = np.lexsort((P["ord"][preds], -ceff[preds])); mt = set()
    for k in order.tolist():
        best, bj = -np.inf, -1
        for q in range(st[k], en[k]):
            j = int(pj[q])
            if j in mt: continue
            if ps[q] > best + 1e-9: best, bj = ps[q], j
        if bj >= 0: mt.add(bj); hit[preds[k]] = True
    return hit
mn = {}
for mode in ("global", "region"):
    cur = {}
    for c in CL:
        P = R["P"][c]; h = greedy(c, mode); o = np.lexsort((P["ord"], -P["conf"])); g = Counter(R["gt_region"][c])
        cur[c] = ClassCurve(conf=P["conf"][o].astype(float), hit=h[o], pred_region=P["region"][o].tolist(),
                            n_gt=sum(g.values()), gt_regions=g, pred_group=P["group"][o].tolist())
    for held in folds:
        res = sweep_groups({c: curve_where(cv, every - held) for c, cv in cur.items()})
        for c, v in res.items():
            for g, gv in v["groups"].items():
                k = f"{g}|{c}"; mn[k] = min(mn.get(k, np.inf), float(gv["threshold"]))
json.dump(mn, open("/workspace/koz3_shared/bot-2/work/merge/min_tau.json", "w"), indent=1)
print({k: v for k, v in sorted(mn.items())})
