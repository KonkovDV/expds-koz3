#. ПОДТВЕРЖДАЮЩИЙ прогон по PREREG_gor_dense_bot5.md (пост-хок -> максимум PACKAGE-ONLY).
"""Основной F|all+cp:add+gate, вторичный FN|all+cp:add+gate, контроль gate (без точек), база = стек A (без гейта и точек).
- s (приор центра) выбран ВЛОЖЕННО в dense_probe.py (внутренний OOF обучения каждого внешнего фолда, cp_s в peaks).
- F: пороги групп стека = пороги контроля gate (sweep_groups на train фолда); порог gor_dense = max train-F1 (ничья -> меньше точек).
- FN: F + удаление точек ближе D к строкам gor стека сцены; D ∈ {30,100,300,1000,inf} по префегу, inf = удалить все точки
  в сцене со строкой gor стека («только сцены без городища в стеке»); D выбирается в каждом внешнем фолде по train-F1
  (ничья -> больший D). Отложенная семья ни в каком выборе не участвует.
- Раскладки: fam10, LOFO (пики analysis/gor_dense/peaks.json), seed5-9 (новый прогон analysis/gor_dense_s59/peaks.json).
- unseen = drop у базы и варианта (группа без строк в train -> порог inf), П6.
Выход: analysis/gor_dense_s59/confirm.json. Env: DOUT (=analysis/gor_dense), D59."""
import os, sys, json, math, collections, time, hashlib
from pathlib import Path
import numpy as np
os.environ['MODE'] = 'none'
_d = (Path(__file__).resolve().parent / 'gor_dense_diag.py').read_text(encoding='utf-8').split("\nif MODE == 'diag':")[0]
exec(compile(_d, 'gor_dense_diag_head', 'exec'))
D59 = Path(os.environ.get('D59', '/workspace/koz3_shared/bot-5/analysis/gor_dense_s59'))
PK2 = json.load(open(D59 / 'peaks.json', encoding='utf-8'))
for s in [int(l[4:]) for l in os.environ.get('LAYS', 'seed5,seed6,seed7,seed8,seed9').split(',') if l.startswith('seed')]:
    nm = f'seed{s}'; layouts[nm] = rand_folds(s)
    assert len(PK2['layouts'][nm]) == 10 and all(set(a['held']) == set(b) for a, b in zip(PK2['layouts'][nm], layouts[nm])), nm
    PK['layouts'][nm] = PK2['layouts'][nm]
V, W_ = 'all+cp', 'add+gate'; DG = [30, 100, 300, 1000, 'scene']
LAY = os.environ.get('LAYS', 'fam10,LOFO,seed5,seed6,seed7,seed8,seed9').split(',')
def frozen_pick(cv, train, taus_c):
    ct = curve_where(cv, train); g = np.asarray(ct.pred_group, dtype=object); dn = g == 'gor_dense'
    tau = np.array([taus_c.get(x, np.inf) for x in g]); ks = (~dn) & (ct.conf >= tau)
    tp0, n0 = int(ct.hit[ks].sum()), int(ks.sum()); dc = ct.conf[dn]; dh = ct.hit[dn]; o = np.argsort(-dc, kind='stable')
    best_t, best_f = np.inf, float(M._f1(tp0, n0 - tp0, ct.n_gt - tp0)) if n0 else 0.0
    ctp, cn = np.cumsum(dh[o]), np.arange(1, len(o) + 1)
    for k in range(len(o)):
        if k + 1 < len(o) and dc[o[k + 1]] == dc[o[k]]: continue
        tp, n = tp0 + int(ctp[k]), n0 + int(cn[k]); f = float(M._f1(tp, n - tp, ct.n_gt - tp))
        if f > best_f + 1e-12: best_f, best_t = f, float(dc[o[k]])
    return best_t, best_f
def held_counts(cv, held, taus_c, t_d, per):
    hc = curve_where(cv, held); g = np.asarray(hc.pred_group, dtype=object)
    tau = np.array([t_d if x == 'gor_dense' else taus_c.get(x, np.inf) for x in g]); kept = hc.conf >= tau
    for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
    for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
def keep_D(row, D): return (not math.isfinite(row[4])) if D == 'scene' else row[4] > D
big = set(json.load(open('/workspace/koz3_shared/bot-10/work10/rw/big_scenes.json'))['big_ge20MB'])
NB = sum(r in big for r in regions); NR = len(regions)
WB, WS = (17 / 30) / (NB / NR), (13 / 30) / ((NR - NB) / NR); WRW = {r: (WB if r in big else WS) for r in regions}
def gf1(per, w=None):
    a = b = g = 0.0
    for r, (x, y_, z) in per.items():
        q = 1.0 if w is None else w.get(r, 0.0); a += q * x; b += q * y_; g += q * z
    return float(M._f1(a, b, g - a))
C = {}; CH = collections.defaultdict(dict)
for nm in LAY:
    folds = layouts[nm]; every = set().union(*folds); per = {'base': {}, 'gate': {}, 'F': {}, 'FN': {}}
    for fr, held in zip(PK['layouts'][nm], folds):
        train = every - held; gp_ = gate_of(train, held)
        for k, gt_ in (('base', None), ('gate', gp_)):
            cv, ids = build_ids([], gt_); eval_cv(cv, ids, train, held, per[k])
        cv, _ = build_ids([], gp_); tc = {g: gv['threshold'] for g, gv in sweep_groups({G: curve_where(cv, train)})[G]['groups'].items()}
        rw = rows_of(fr, V, gp_)
        cv, _ = build_ids(rw, gp_); t_d, f = frozen_pick(cv, train, tc); held_counts(cv, held, tc, t_d, per['F'])
        CH[nm].setdefault('F', []).append([None, t_d, fr.get('cp_s', {}).get('all')])
        best = None
        for D in DG:
            cv, _ = build_ids([r_ for r_ in rw if keep_D(r_, D)], gp_); t_d, f = frozen_pick(cv, train, tc)
            if best is None or f >= best[2] - 1e-12: best = (D, t_d, f, cv)
        held_counts(best[3], held, tc, best[1], per['FN']); CH[nm].setdefault('FN', []).append([best[0], best[1]])
    assert gsc(per['base']) == gsc(gor_counts(folds, None)), f'{nm}: base mismatch'
    C[nm] = per; log(nm, {k: gsc(v) for k, v in per.items()})
def full(per): return {**base_full, G: per}
HEAVY = ('kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha')
def perclass(a, b, w=None):
    _, pa = score_counts(a, None, w); _, pb = score_counts(b, None, w); den = sum(M.CLASS_WEIGHTS.get(c, 0) for c in pa if pa[c]['n_gt'] > 0)
    return {c: M.CLASS_WEIGHTS.get(c, 0) * (pa[c]['f1'] - pb[c]['f1']) / den for c in HEAVY}
def dl(nm, k, ref='base', w=None): return 15 * (gf1(C[nm][k], w) - gf1(C[nm][ref], w)) / 100
assert abs(M.CLASS_WEIGHTS[G] - 15) < 1e-9 and abs(sum(M.CLASS_WEIGHTS.values()) - 100) < 1e-9
fams = collections.defaultdict(list)
for r in sorted(regions): fams[site_family(r)].append(r)
names = sorted(fams); S59 = [l for l in LAY if l.startswith('seed')]
def boot_seeds(ka, kb, w0=None, n=1000):
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        w = {r: float(q) * (1.0 if w0 is None else w0[r]) for f, q in zip(names, pick) if q for r in fams[f]}
        out.append(np.mean([15 * (gf1(C[s][ka], w) - gf1(C[s][kb], w)) / 100 for s in S59]))
    v = np.asarray(out); return dict(mean=float(v.mean()), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)), p_gt0=float((v > 0).mean()), n=n)
R = {'prereg': '/workspace/koz3_shared/bot-10/PREREG_gor_dense_bot5.md', 'prerun': str(D59 / 'prerun.json'),
     'prerun_sha256': hashlib.sha256((D59 / 'prerun.json').read_bytes()).hexdigest(),
     'peaks_sha256': {'fam10_LOFO': hashlib.sha256((DOUT / 'peaks.json').read_bytes()).hexdigest(), 'seed5_9': hashlib.sha256((D59 / 'peaks.json').read_bytes()).hexdigest()},
     'unseen_mode': 'drop for base and variant (taus.get(group, inf)) — P6', 'A_honest10_drop': A, 'D_grid': DG, 'status': 'post-hoc -> max PACKAGE-ONLY',
     'chosen': CH, 'cp_s_chosen_seed5_9': PK2.get('cp_s_chosen'), 'variants': {}}
for k in ('gate', 'F', 'FN'):
    vf = full(C['fam10'][k]); o = {}
    o['fam10'] = dict(delta_exact=score_counts(vf)[0] - A, delta_new34=score_counts(vf, new)[0] - score_counts(base_full, new)[0],
                      delta_optical=score_counts(vf, optical)[0] - score_counts(base_full, optical)[0],
                      per_heavy_class=perclass(vf, base_full), bootstrap=paired_bootstrap_counts(vf, base_full, site_family, n=1000, seed=0),
                      counts=gsc(C['fam10'][k]), counts_base=gsc(C['fam10']['base']))
    o['LOFO'] = dl('LOFO', k) if 'LOFO' in C else None; o['seeds5_9'] = [dl(s, k) for s in S59]; o['seeds5_9_mean'] = float(np.mean(o['seeds5_9']))
    o['seeds5_9_boot'] = boot_seeds(k, 'base'); o['vs_gate_control'] = {'fam10_exact': score_counts(vf)[0] - score_counts(full(C['fam10']['gate']))[0], 'LOFO': dl('LOFO', k, 'gate') if 'LOFO' in C else None, 'seeds5_9': [dl(s, k, 'gate') for s in S59]}
    o['reweighted_big'] = dict(n_big=NB, N=NR, w_big=WB, w_small=WS, fam10_full=score_counts(vf, None, WRW)[0] - score_counts(base_full, None, WRW)[0],
                               fam10_per_heavy=perclass(vf, base_full, WRW), LOFO=dl('LOFO', k, w=WRW) if 'LOFO' in C else None, seeds5_9=[dl(s, k, w=WRW) for s in S59])
    # бутстреп перевзвешенного fam10: семьи × вес состава (как rw_paired.py)
    rng = np.random.default_rng(0); bb = []
    for _ in range(1000):
        pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
        w = {r: float(q) * WRW[r] for f, q in zip(names, pick) if q for r in fams[f]}
        bb.append(score_counts(vf, None, w)[0] - score_counts(base_full, None, w)[0])
    o['reweighted_big']['boot_fam10'] = dict(p_gt0=float((np.asarray(bb) > 0).mean()), p5=float(np.percentile(bb, 5)), p95=float(np.percentile(bb, 95)))
    R['variants'][k] = o; log(k, json.dumps({q: o[q] for q in ('LOFO', 'seeds5_9', 'seeds5_9_mean')}), {q: o['fam10'][q] for q in ('delta_exact', 'delta_new34', 'delta_optical')})
F_ = R['variants']['F']; f10 = F_['fam10']; sd = F_['seeds5_9']
crit = {'seeds5_9_mean>0_and_>=4of5': [F_['seeds5_9_mean'], sd, F_['seeds5_9_mean'] > 0 and sum(x > 0 for x in sd) >= 4],
        'P1_exact>=0.002_and_LOFO>=0.002': [f10['delta_exact'], F_['LOFO'], f10['delta_exact'] >= 0.002 and (F_['LOFO'] or 0) >= 0.002],
        'P2_family_boot_p>=0.85': [f10['bootstrap']['p_gt0'], f10['bootstrap']['p_gt0'] >= 0.85],
        'P4_heavy_each_drop<=0.0005': [f10['per_heavy_class'], all(v >= -0.0005 for v in f10['per_heavy_class'].values())],
        'P4_new34>=-0.001': [f10['delta_new34'], f10['delta_new34'] >= -0.001],
        'P6_same_unseen_mode': ['drop/drop', True]}
crit['PRIMARY_CONFIRMED'] = all(v[-1] for v in crit.values())
dfn = [a - b for a, b in zip(R['variants']['FN']['seeds5_9'], sd)]; bfn = boot_seeds('FN', 'F')
crit_fn = {'FN_minus_F_seeds5_9_mean>=0.002': [float(np.mean(dfn)), dfn, float(np.mean(dfn)) >= 0.002], 'P>=0.85_seeds5_9': [bfn, bfn['p_gt0'] >= 0.85]}
crit_fn['FN_REPLACES_F'] = crit_fn['FN_minus_F_seeds5_9_mean>=0.002'][-1] and crit_fn['P>=0.85_seeds5_9'][-1]
R['criteria_primary'] = crit; R['criteria_FN'] = crit_fn
R['verdict'] = ('PACKAGE-ONLY (post-hoc ceiling), pending paired submission-order run on package' if crit['PRIMARY_CONFIRMED'] else 'NOT CONFIRMED')
log('criteria', json.dumps(crit, default=float)); log('FN', json.dumps(crit_fn, default=float)); log('verdict', R['verdict'])
json.dump(R, open(D59 / os.environ.get('CONF_OUT', 'confirm.json'), 'w'), ensure_ascii=False, indent=1, default=float)
