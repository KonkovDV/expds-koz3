"""Что теряется на стороне GT: честный k10 (как localize_exact.py, вариант A), для каждого GT-объекта — попал ли он в TP
после порогов и достижим ли вообще (есть пара с любым кандидатом до порога). Группы: класс × markup_type GT, регион.
Потолок группы = прирост итогового скора, если все её FN станут TP (oracle, FP не меняются); 'reach' — только достижимые."""
import sys, os, json, time, resource, pickle, gc
resource.setrlimit(resource.RLIMIT_AS, (3_800_000_000, 3_800_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np, copy
from collections import Counter, defaultdict
from common import *
from expds.metric import ClassCurve, sweep_groups, curve_where
from expds import metric as M
mod = M.load_official(METRIC); calc = mod.F1Calculator()
T = pickle.load(open(W/'work/table.pkl', 'rb'))
CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}
cls, conf, reg, grp = T['cls'], T['conf'], T['reg'], T['grp']
for k in ('area','perim','cx','cy','nv','bw','bh','mk','gx','gy'): T.pop(k, None)
PL0 = {}; REACH = {c: set() for c in CL}
for (pc, gc_), (P, J, S) in T['pairs'].items():
    d = {}; o = np.lexsort((J, P))
    for p, j, s in zip(P[o], J[o], S[o]): d.setdefault(int(p), []).append((int(j), float(s)))
    PL0[(pc, gc_)] = d; REACH[gc_].update(int(j) for j in J)
del T['pairs']; gc.collect()
GREG = {c: Counter(T['greg'][c]) for c in CL}
# markup GT в том же порядке, что в prep.py (M._collect на том же файле)
gt = load(W/'gt_train_v2.geojson'); gts = M._collect(mod, calc, gt, True); del gt; gc.collect()
GMK = {c: [x[4] if len(x) > 4 else x[3] for x in gts.get(c, [])] for c in CL}
for c in CL:
    assert len(GMK[c]) == T['n_gt'][c], c
    assert [x[2] for x in gts.get(c, [])] == list(T['greg'][c]), c
del gts; gc.collect()
grp_s = grp.astype(str)
def match(order_idx, c, PL, matched=None):
    hit = np.zeros(len(order_idx), bool); matched = set() if matched is None else matched
    for r, p in enumerate(order_idx):
        lst = PL.get((CL[cls[p]], c), {}).get(int(p))
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in lst:
            if j in matched: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: matched.add(bj); hit[r] = True
    return hit
CV = {}
for c in CL:
    ii = np.flatnonzero(cls == cid[c]); o = ii[np.argsort(-conf[ii], kind='stable')]
    CV[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=match(o, c, PL0), pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                       gt_regions=GREG[c], pred_group=list(grp[o]))
gt_regions = set().union(*[set(GREG[c]) for c in CL])
regions = {r for i in range(len(CL)) for r in np.unique(reg[cls == i])} | gt_regions
folds = family_folds(regions, site_family, 10)
every = set().union(*folds); regarr = reg.astype(object)
cnt = {}; MATCHED = {c: set() for c in CL}; nkept = 0; kept_by_reg = Counter()
for held in folds:
    res = sweep_groups({c: curve_where(v, every - held) for c, v in CV.items()})
    tau = {(g, cid[c]): gv['threshold'] for c, v in res.items() for g, gv in v['groups'].items()}
    hi = np.flatnonzero(np.isin(regarr, list(held)) & (cls >= 0))
    th = np.array([tau.get((grp_s[i], int(cls[i])), np.inf) for i in hi])
    keep = hi[conf[hi] >= th]; nkept += len(keep); kept_by_reg.update(reg[keep].tolist())
    for c in CL:
        per = cnt.setdefault(c, {})
        for r in held:
            n = GREG[c].get(r, 0)
            if n: per.setdefault(r, [0, 0, 0])[2] += int(n)
        ii = keep[cls[keep] == cid[c]]; o = ii[np.argsort(-conf[ii], kind='stable')]
        for p, h in zip(o, match(o, c, PL0, MATCHED[c])): per.setdefault(reg[p], [0, 0, 0])[0 if h else 1] += 1
base, per = score_counts(cnt)
print('base k10 %.6f kept %d' % (base, nkept), {c: (per[c]['tp'], per[c]['fp'], round(per[c]['f1'], 4)) for c in CL}, flush=True)
def oracle(sel):  # sel(c, j) -> bool; FN из выбранных становятся TP
    c2 = copy.deepcopy(cnt)
    for c in CL:
        for j, (r, mk) in enumerate(zip(T['greg'][c], GMK[c])):
            if j not in MATCHED[c] and sel(c, j): c2[c][r][0] += 1
    return score_counts(c2)[0] - base
rows = []
for c in CL:
    for mk in sorted(set(GMK[c])):
        J = [j for j, m in enumerate(GMK[c]) if m == mk]
        tp = sum(j in MATCHED[c] for j in J); re = sum(j in REACH[c] for j in J)
        rows.append(dict(cls=c, markup=mk, n_gt=len(J), tp=tp, recall=round(tp / len(J), 4), reachable=re, reach_share=round(re / len(J), 4),
                         oracle_all=round(oracle(lambda cc, j, mk=mk, c=c: cc == c and GMK[cc][j] == mk), 6),
                         oracle_reach=round(oracle(lambda cc, j, mk=mk, c=c: cc == c and GMK[cc][j] == mk and j in REACH[cc]), 6)))
        print(rows[-1], flush=True)
regrows = []
for r in sorted(gt_regions):
    n = sum(GREG[c].get(r, 0) for c in CL)
    tp = sum(1 for c in CL for j, rr in enumerate(T['greg'][c]) if rr == r and j in MATCHED[c])
    re = sum(1 for c in CL for j, rr in enumerate(T['greg'][c]) if rr == r and j in REACH[c])
    if tp == 0 or re == 0:
        regrows.append(dict(region=r, n_gt=n, tp=tp, reachable=re, kept_preds=kept_by_reg.get(r, 0),
                            classes={c: GREG[c].get(r, 0) for c in CL if GREG[c].get(r, 0)},
                            oracle_all=round(oracle(lambda cc, j, r=r: T['greg'][cc][j] == r), 6)))
regrows.sort(key=lambda x: -x['oracle_all'])
for x in regrows: print('REG', x, flush=True)
kr = np.array([kept_by_reg.get(r, 0) for r in sorted(regions)])
json.dump(dict(base=base, kept=nkept, kept_per_region=dict(p50=float(np.median(kr)), p90=float(np.percentile(kr, 90)), max=int(kr.max()), n_regions=len(kr)),
               groups=rows, zero_regions=regrows, per_class={c: per[c] for c in CL}), open('lost.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
