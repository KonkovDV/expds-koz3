"""ORSP диагностика (bot-6): потери курганов/arch/fort на Or (or+sp_or) на честной базе A.
Источник: /workspace/rerank/work/table.pkl; логика kept = bot-4/work/gap/gap.py (honest-10 canon, unseen=drop)."""
import sys, json, pickle, collections, time
import numpy as np
from pathlib import Path
ROOT = Path('/workspace/expds-koz3'); sys.path.insert(0, str(ROOT))
from expds import metric as M
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, sweep_groups, curve_where, _f1, CLASS_WEIGHTS
from expds.naming import site_family
OUT = Path(__file__).parent
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; W = {c: CLASS_WEIGHTS[c] for c in CL}
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py'))
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
gmk = {c: [] for c in CL}
for f in gt['features']:
    p = f.get('properties') or {}; c = mod.normalize_class_name(p.get('class_name', 'unknown'))
    if c in gmk and f.get('geometry'): gmk[c].append(p.get('markup_type'))
del gt
for c in CL: assert len(gmk[c]) == T['n_gt'][c]
boxes = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))
cls, conf, reg, grp, mk = T['cls'], T['conf'], T['reg'], T['grp'], T['mk']

def greedy(c, cand, key=None):
    P, J, S = T['pairs'][key or (c, c)]
    by = collections.defaultdict(list)
    for a, b, s in zip(P, J, S): by[a].append((b, s))
    matched = {}; used = set()
    for i in cand:
        lst = by.get(i)
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in sorted(lst):
            if j in used: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); matched[i] = bj
    return matched
order = {}; curves = {}
for k, c in enumerate(CL):
    idx = np.flatnonzero(cls == k); o = idx[np.argsort(-conf[idx], kind='stable')]; order[c] = o
    m = greedy(c, o); hit = np.fromiter((i in m for i in o), bool, len(o))
    curves[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                           gt_regions=collections.Counter(T['greg'][c]), pred_group=list(grp[o]))
regions = {r for cv in curves.values() for r in cv.pred_region} | {r for cv in curves.values() for r in cv.gt_regions}
folds = family_folds(regions, site_family, 10)
counts, info = honest_counts(curves, folds); A = score_counts(counts)[0]
print('honest A', A, flush=True)
kept = np.zeros(len(cls), bool); every = set().union(*folds); tau_f = []
for held in folds:
    res = sweep_groups({c: curve_where(cv, every - held) for c, cv in curves.items()})
    tau_f.append({c: {g: gv['threshold'] for g, gv in res.get(c, {'groups': {}})['groups'].items()} for c in CL})
    for c in CL:
        taus = tau_f[-1][c]; o = order[c]; inh = np.isin(reg[o], list(held))
        tau = np.array([taus.get(g, np.inf) for g in grp[o]]); kept[o[inh & (conf[o] >= tau)]] = True
# exact rematch on kept
gm = {c: greedy(c, order[c][kept[order[c]]]) for c in CL}
tot = {}
for c in CL:
    n = T['n_gt'][c]; tp = len(gm[c]); fp = int(kept[order[c]].sum()) - tp; tot[c] = [tp, fp, n - tp]
sc = lambda t: sum(W[c] * float(_f1(*t[c])) for c in CL) / 100
print('A exact rematch', sc(tot), flush=True)
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
def other(c): return KUR[1 - KUR.index(c)]
out = {'A_common': A, 'A_rematch': sc(tot), 'tot': tot, 'rows': {}, 'fp_by_group': {}, 'tau_by_fold': {}}
for c in ['kurgany_povrezhdennye', 'kurgany_tselye', 'arkhitektury', 'fortifikatsii', 'object_poly', 'gorodishcha']:
    n = T['n_gt'][c]; gtm = np.array(gmk[c]); tpg = np.zeros(n, bool); tpg[list(gm[c].values())] = True
    P, J, S = T['pairs'][(c, c)]
    same_any = collections.defaultdict(list)
    for a, b in zip(P, J): same_any[b].append(a)
    cross_any = collections.defaultdict(list)
    if c in KUR:
        P2, J2, _ = T['pairs'][(other(c), c)]   # (pred class, gt class)
        for a, b in zip(P2, J2): cross_any[b].append(a)
    rows = collections.Counter(); bygrp = collections.Counter()
    for j in range(n):
        m_ = gtm[j]
        if m_ not in ('or', 'sp_or'): continue
        lid = 'lidar' if T['greg'][c][j] in boxes else 'nolidar'
        key = (m_, lid)
        rows[key + ('n',)] += 1
        if tpg[j]: rows[key + ('tp',)] += 1; continue
        S_ = same_any.get(j, []); X_ = cross_any.get(j, [])
        Sk = [i for i in S_ if kept[i]]; Xk = [i for i in X_ if kept[i]]
        if Sk: cat = 'competition'          # same-class kept candidate exists, matched elsewhere
        elif Xk: cat = 'confusion_kept'      # only wrong-class kept output at the place
        elif S_: cat = 'threshold'           # same-class candidate exists, cut by τ
        elif X_: cat = 'confusion_cut'       # only wrong-class candidate, also cut
        else: cat = 'hole'
        rows[key + (cat,)] += 1
        if cat == 'threshold':
            best = max(S_, key=lambda i: conf[i]); bygrp[(str(grp[best]), str(mk[best]))] += 1
        if cat == 'confusion_kept':
            best = max(Xk, key=lambda i: conf[i]); bygrp[('X:' + str(grp[best]), str(mk[best]))] += 1
    out['rows'][c] = {'|'.join(k): v for k, v in sorted(rows.items())}
    out['rows'][c + ':grp'] = {'|'.join(k): v for k, v in sorted(bygrp.items())}
    # FP by group (kept, unmatched)
    o = order[c][kept[order[c]]]; fpc = collections.Counter(); tpc = collections.Counter()
    for i in o:
        (tpc if i in gm[c] else fpc)[(str(grp[i]), str(mk[i]))] += 1
    out['fp_by_group'][c] = {f'{g}|{m_}': [tpc[(g, m_)], fpc[(g, m_)]] for (g, m_) in sorted(set(fpc) | set(tpc))}
    out['tau_by_fold'][c] = [{g: t for g, t in tf[c].items() if g.startswith('sp') or g == 'or'} for tf in tau_f]
# ceilings (Δ score vs rematch A) per category, all Or GT of class
def ceil(c, k, dfp=0, c2=None, dfp2=0):
    t = {x: list(v) for x, v in tot.items()}; t[c][0] += k; t[c][2] -= k; t[c][1] += dfp
    if c2: t[c2][1] += dfp2
    return sc(t) - sc(tot)
cl = {}
for c in KUR + ['arkhitektury', 'fortifikatsii']:
    r = out['rows'][c]
    for cat in ['hole', 'threshold', 'confusion_kept', 'confusion_cut', 'competition']:
        k = sum(v for kk, v in r.items() if kk.endswith('|' + cat))
        cl[f'{c}|{cat}'] = dict(k=k, ceil_free=ceil(c, k))
    # confusion_kept relabel: c gets +k TP, other class loses k FP (if those kept were FP) -- upper bound
    if c in KUR:
        k = cl[f'{c}|confusion_kept']['k']; cl[f'{c}|confusion_kept']['ceil_relabel'] = ceil(c, k, 0, other(c), -k)
out['ceilings'] = cl
json.dump(out, open(OUT / 'res_diag.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out['rows'], ensure_ascii=False, indent=0)); print(json.dumps(cl, indent=0)); print(json.dumps(out['fp_by_group'], indent=0))
