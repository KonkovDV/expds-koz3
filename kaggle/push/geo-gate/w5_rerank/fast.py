"""Exact replica of metric.build_curves on precomputed pairs (any rescoring / class reassignment)."""
import numpy as np, pickle
from collections import Counter
from common import W
from expds.metric import ClassCurve
T = pickle.load(open(W/'work/table.pkl','rb'))
CL = T['CL']; cid = {c:i for i,c in enumerate(CL)}
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
# per (orig pred class, gt class): dict pred -> (js sorted, ss)
PL = {}
for (pc, gc), (P, J, S) in T['pairs'].items():
    d = {}
    o = np.lexsort((J, P))
    for p, j, s in zip(P[o], J[o], S[o]): d.setdefault(int(p), []).append((int(j), float(s)))
    PL[(pc, gc)] = d
GREG = {c: Counter(T['greg'][c]) for c in CL}

def curves(score=None, assign=None, keep=None):
    """score: per-candidate confidence (default original); assign: per-candidate class id (default original);
    keep: bool mask of candidates emitted at all."""
    conf = T['conf'] if score is None else score
    cls = T['cls'] if assign is None else assign
    out = {}
    for c in CL:
        m = cls == cid[c]
        if keep is not None: m &= keep
        idx = np.flatnonzero(m)
        o = idx[np.argsort(-conf[idx], kind='stable')]
        hit = np.zeros(len(o), bool); matched = set()
        for r, p in enumerate(o):
            oc = CL[T['cls'][p]]
            lst = PL.get((oc, c), {}).get(int(p))
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r] = True
        out[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(T['reg'][o]), n_gt=T['n_gt'][c],
                            gt_regions=GREG[c], pred_group=list(T['grp'][o]))
    return out
