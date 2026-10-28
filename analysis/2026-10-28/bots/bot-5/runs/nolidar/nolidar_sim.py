"""(1) цена одного TP/FP по классам в честном балле A; (2) дубль спутниковых кругов курганов во второй класс курганов
(путаница повр/целые) — честный-10 + бутстреп по семьям. Всё на table.pkl, официальные пары."""
import sys, json, pickle, collections, time, copy
import numpy as np
from pathlib import Path
ROOT = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(ROOT))
from expds import metric as M
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, paired_bootstrap_counts
from expds.naming import site_family
OUT = Path('/workspace/koz3_shared/bot-5/analysis/nolidar')
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; cls, conf, reg, grp = T['cls'], T['conf'], T['reg'], T['grp']
boxes = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
def build(c, extra=()):
    """extra: список (src_class, rows, group_name) — строки другого класса как кандидаты класса c."""
    k = CL.index(c); idx = np.flatnonzero(cls == k)
    items = [(i, c, g) for i, g in zip(idx, grp[idx])]
    for sc, rows, gname in extra: items += [(i, sc, gname) for i in rows]
    cf = np.array([conf[i] for i, _, _ in items]); o = np.argsort(-cf, kind='stable'); items = [items[j] for j in o]
    by = {}
    for sc in {s for _, s, _ in items}:
        P, J, S = T['pairs'][(sc, c)]; d = collections.defaultdict(list)
        for a, b, s in zip(P, J, S): d[a].append((b, s))
        by[sc] = d
    used = set(); hit = np.zeros(len(items), bool)
    for n, (i, sc, _) in enumerate(items):
        lst = by[sc].get(i)
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in sorted(lst):
            if j in used: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); hit[n] = True
    return ClassCurve(conf=cf[o].astype(np.float64), hit=hit, pred_region=[reg[i] for i, _, _ in items], n_gt=T['n_gt'][c],
                      gt_regions=collections.Counter(T['greg'][c]), pred_group=[g for _, _, g in items])
base = {c: build(c) for c in CL}
regions = {r for cv in base.values() for r in cv.pred_region} | {r for cv in base.values() for r in cv.gt_regions}
folds = family_folds(regions, site_family, 10)
nol = {r for r in regions if r not in boxes}
cA, _ = honest_counts(base, folds); sA = score_counts(cA)[0]; sAn = score_counts(cA, nol)[0]
out = {'A_all': sA, 'A_nolidar_regions': sAn}
# (1) цена TP / FP: добавить к первому участку класса
tot = {c: [sum(v[i] for v in cA[c].values()) for i in range(3)] for c in CL}
out['A_totals_tp_fp_gt'] = tot
val = {}
for c in CL:
    for kind in ('tp', 'fp'):
        for dn in (10, 100):
            cc = copy.deepcopy(cA); r = next(iter(cc[c]))
            if kind == 'tp': cc[c][r][0] += dn
            else: cc[c][r][1] += dn
            val[f'{c}|+{dn}{kind}'] = score_counts(cc)[0] - sA
out['marginal'] = val
print(json.dumps(val, indent=0), flush=True)
# (2) путаница классов: спутниковые круги одного класса курганов дублируются в другой (отдельная группа, свой порог)
SPG = ('sp_focal', 'sp_focal2')
def rows_of(c, groups): k = CL.index(c); return np.flatnonzero((cls == k) & np.isin(grp, groups))
variants = {
  'povr_sp_to_tselye': {'kurgany_tselye': [('kurgany_povrezhdennye', rows_of('kurgany_povrezhdennye', SPG), 'x_sp_p2t')]},
  'tselye_sp_to_povr': {'kurgany_povrezhdennye': [('kurgany_tselye', rows_of('kurgany_tselye', SPG), 'x_sp_t2p')]},
}
variants['both'] = {**variants['povr_sp_to_tselye'], **variants['tselye_sp_to_povr']}
for name, ex in variants.items():
    cur = dict(base)
    for c, e in ex.items(): cur[c] = build(c, e)
    cB, _ = honest_counts(cur, folds)
    o = {'all': score_counts(cB)[0], 'delta_all': score_counts(cB)[0] - sA,
         'delta_nolidar': score_counts(cB, nol)[0] - sAn,
         'boot': paired_bootstrap_counts(cB, cA, site_family, 1000),
         'tot': {c: [sum(v[i] for v in cB[c].values()) for i in range(3)] for c in KUR}}
    for sd in range(5):
        fs = family_folds(regions, site_family, 10, seed=sd) if 'seed' in family_folds.__code__.co_varnames else None
        if fs is None: break
        a_, _ = honest_counts(base, fs); b_, _ = honest_counts(cur, fs)
        o[f'seed{sd}'] = score_counts(b_)[0] - score_counts(a_)[0]
    out[name] = o; print(name, json.dumps(o)[:600], flush=True)
json.dump(out, open(OUT / 'nolidar_sim.json', 'w'), ensure_ascii=False, indent=1)
