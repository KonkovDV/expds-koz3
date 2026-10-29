# bot-10 work14. Парный замер Package+F − Package для плотного локализатора городищ bot-5 (F|all+cp:add+gate)
# поверх ПАКЕТА с D-FINE R. Машинерия bot-5 (только чтение, exec головы gor_dense_diag.py, как prereg_confirm.py).
# Пакет для класса gorodishcha: FR1h (городища исключены), ORGATE G (только arch), twins (только курганы) класс не трогают;
# D-FINE R: строки городищ групп sp_or/sp_focal/sp_arch/sp_focal2 на 132 участках D-FINE удаляются, добавляются круги
# sp_dfine (work14/dfine_gor.json). Метрика = взвешенное среднее F1 классов => Δ(Package+F − Package) = 15/100·ΔF1_gor.
# Варианты: A (стек A), AF (= F bot-5 над A, повтор), P (пакет), PF (пакет + точки dense; гейт-умножение sp_or при R пусто,
# p сцены остаётся только в conf точки), PgD (эксплор.: гейт умножает conf кругов sp_dfine), PFgD (эксплор.).
# Пороги: вложенно, sweep_groups на train фолда; F: пороги стека = пороги контроля, τ gor_dense = max train-F1 (frozen_pick bot-5).
# Чувствительность SUB: τ sp_dfine gor = 0,68847 (порог сдачи, bot-7/work6/on/make_on.py) вместо вложенного.
import os, sys, json, math, collections, time, hashlib
from pathlib import Path
import numpy as np
B5 = Path('/workspace/koz3_shared/bot-5/repo/scripts/gor_dense'); W = Path(__file__).resolve().parent
os.environ['MODE'] = 'none'
_d = (B5 / 'gor_dense_diag.py').read_text(encoding='utf-8').split("\nif MODE == 'diag':")[0]
_myfile = __file__; __file__ = str(B5 / 'gor_dense_diag.py')
exec(compile(_d, 'gor_dense_diag_head', 'exec'))
__file__ = _myfile
D59 = Path('/workspace/koz3_shared/bot-5/analysis/gor_dense_s59'); PK2 = json.load(open(D59 / 'peaks.json', encoding='utf-8'))
LAY = os.environ.get('LAYS', 'fam10,LOFO,seed0,seed1,seed2,seed3,seed4,seed5,seed6,seed7,seed8,seed9').split(',')
for nm in LAY:
    if nm.startswith('seed') and int(nm[4:]) >= 5:
        layouts[nm] = rand_folds(int(nm[4:])); PK['layouts'][nm] = PK2['layouts'][nm]
    assert len(PK['layouts'][nm]) == len(layouts[nm]) and all(set(a['held']) == set(b) for a, b in zip(PK['layouts'][nm], layouts[nm])), nm
# ---- D-FINE R ----
DG_ = json.load(open(W / 'dfine_gor.json', encoding='utf-8')); COV = set(DG_['report']['regions'])
DROP = {'sp_or', 'sp_focal', 'sp_arch', 'sp_focal2'}
keep_rows = [int(i) for i in gor_rows if not (grp[i] in DROP and reg[i] in COV)]
dropped = len(gor_rows) - len(keep_rows)
DF0 = N0 + 10_000_000; DFP = {}; DFC = []; DFR = []
for k, c in enumerate(DG_['circles']):
    poly = mod._to_utm(shapely.Polygon(c['coords']), calc.input_crs_mode); r = c['region']; out = []  # EPSG:3857 -> UTM, как метрика (_collect_by_class)
    for j in np.flatnonzero(greg == r):
        g = GG[j]; gg = g.buffer(calc.r_tolerance) if g.geom_type == 'Point' else g
        if not poly.intersects(gg): continue
        ok, s = calc._match_quality(poly, g)
        if ok: out.append((int(j), float(s)))
    DFP[DF0 + k] = out; DFC.append(float(c['conf'])); DFR.append(r)
log('R: dropped stack gor rows', dropped, 'kept', len(keep_rows), 'dfine circles', len(DFC), 'with OK pair', sum(1 for v in DFP.values() if v), 'cov regions', len(COV), 'SCNS<=COV', SCNS <= COV)
def build(pkg, rows, gate_p, gate_dfine=False):
    if not pkg: return build_ids(rows, gate_p)[0]
    ids = list(keep_rows); cf = list(conf[keep_rows].astype(float)); rg = list(reg[keep_rows]); gp = list(grp[keep_rows])
    if gate_p is not None:   # как build_ids bot-5: умножаются только sp_or-строки; при R их на участках D-FINE нет
        for n, i in enumerate(ids):
            if mk[i] == 'sp_or' and rg[n] in gate_p: cf[n] = cf[n] * gate_p[rg[n]]
    for k in range(len(DFC)):
        c = DFC[k]; g = 'sp_dfine'
        if gate_dfine and gate_p is not None and DFR[k] in gate_p: c = c * gate_p[DFR[k]]; g = 'sp_dfine_gated'
        ids.append(DF0 + k); cf.append(c); rg.append(DFR[k]); gp.append(g)
    extra = {}
    for k, row in enumerate(rows):
        r, c, pr = row[:3]; nid = N0 + k; extra[nid] = pr; ids.append(nid); cf.append(c); rg.append(r); gp.append('gor_dense')
    cf = np.array(cf); o = np.argsort(-cf, kind='stable'); ids_o = [ids[k] for k in o]
    hit = greedy_hits(ids_o, lambda i: DFP[i] if i >= DF0 else (extra[i] if i >= N0 else pairs_by.get(i, [])))
    return ClassCurve(conf=cf[o], hit=hit, pred_region=[rg[k] for k in o], n_gt=T['n_gt'][G],
                      gt_regions=collections.Counter(T['greg'][G]), pred_group=[gp[k] for k in o])
def taus_of(cv, train, fix=None):
    t = {g: gv['threshold'] for g, gv in sweep_groups({G: curve_where(cv, train)})[G]['groups'].items()}
    if fix: t.update(fix)
    return t
def frozen_pick(cv, train, taus_c):   # = prereg_confirm.py bot-5
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
SUBFIX = {'sp_dfine': 0.68847}
KEYS = ['A', 'AF', 'P', 'PF', 'PgD', 'PFgD', 'Psub', 'PFsub']
C = {}; CH = collections.defaultdict(lambda: collections.defaultdict(list)); t0_ = time.time()
for nm in LAY:
    folds = layouts[nm]; every = set().union(*folds); per = {k: {} for k in KEYS}
    for fr, held in zip(PK['layouts'][nm], folds):
        train = every - held; gp_ = gate_of(train, held); rw = rows_of(fr, 'all+cp', gp_)
        for kb, kv, pkg, gd, fix in (('A', 'AF', False, False, None), ('P', 'PF', True, False, None), ('PgD', 'PFgD', True, True, None), ('Psub', 'PFsub', True, False, SUBFIX)):
            # контроль: для A — гейт bot-5 (как prereg_confirm: пороги стека F = пороги контроля gate); база A без гейта отдельно
            if kb == 'A':
                cv0 = build(False, [], None); tb = taus_of(cv0, train); held_counts(cv0, held, tb, np.inf, per['A'])
                cvc = build(False, [], gp_); tc = taus_of(cvc, train)
            else:
                cvc = build(True, [], gp_ if gd else None, gd); tc = taus_of(cvc, train, fix); held_counts(cvc, held, tc, np.inf, per[kb])
            cv = build(pkg, rw, gp_, gd); t_d, f = frozen_pick(cv, train, tc); held_counts(cv, held, tc, t_d, per[kv])
            CH[nm][kv].append([t_d, tc.get('sp_dfine'), tc.get('sp_dfine_gated')])
    assert gsc(per['A']) == gsc(gor_counts(folds, None)), f'{nm}: A mismatch'
    C[nm] = per; log(nm, f'{time.time()-t0_:.0f}s', {k: gsc(v) for k, v in per.items()})
    json.dump({'C': C, 'chosen': CH}, open(W / 'pkg_gor_counts.json', 'w'), ensure_ascii=False, default=float)
# ---- статистика ----
def full(per): return {**base_full, G: per}
HEAVY = ('kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha')
def perclass(a, b):
    _, pa = score_counts(a); _, pb = score_counts(b); den = sum(M.CLASS_WEIGHTS.get(c, 0) for c in pa if pa[c]['n_gt'] > 0)
    return {c: M.CLASS_WEIGHTS.get(c, 0) * (pa[c]['f1'] - pb[c]['f1']) / den for c in HEAVY}
def gf1(per):
    a = sum(v[0] for v in per.values()); b = sum(v[1] for v in per.values()); g = sum(v[2] for v in per.values()); return float(M._f1(a, b, g - a))
def dl(nm, k, ref): return 15 * (gf1(C[nm][k]) - gf1(C[nm][ref])) / 100
assert abs(M.CLASS_WEIGHTS[G] - 15) < 1e-9 and abs(sum(M.CLASS_WEIGHTS.values()) - 100) < 1e-9
SEEDS = [l for l in LAY if l.startswith('seed')]
PAIRS = [('PF', 'P'), ('AF', 'A'), ('P', 'A'), ('PF', 'A'), ('PFgD', 'PgD'), ('PgD', 'P'), ('PFgD', 'P'), ('PFsub', 'Psub')]
R = {'definition': 'bot-10/work14/pkg_gor.py; П1–П6 EXPERIMENTS.md; unseen=drop у всех (taus.get(g, inf))', 'A_honest10_drop': A,
     'R_dropped_stack_gor_rows': dropped, 'R_kept_stack_gor_rows': len(keep_rows), 'dfine_circles': len(DFC), 'dfine_regions': len(COV),
     'inputs_sha256': {'dfine_gor.json': hashlib.sha256((W / 'dfine_gor.json').read_bytes()).hexdigest(),
                       'peaks_fam10_LOFO_s0_4': hashlib.sha256((DOUT / 'peaks.json').read_bytes()).hexdigest(),
                       'peaks_s5_9': hashlib.sha256((D59 / 'peaks.json').read_bytes()).hexdigest()},
     'counts_fam10': {k: gsc(v) for k, v in C['fam10'].items()}, 'pairs': {}, 'chosen_fam10': CH['fam10']}
for a, b in PAIRS:
    va, vb = full(C['fam10'][a]), full(C['fam10'][b]); o = {}
    o['fam10_exact'] = score_counts(va)[0] - score_counts(vb)[0]
    o['new34'] = score_counts(va, new)[0] - score_counts(vb, new)[0]; o['optical'] = score_counts(va, optical)[0] - score_counts(vb, optical)[0]
    o['per_heavy'] = perclass(va, vb); o['boot'] = paired_bootstrap_counts(va, vb, site_family, n=1000, seed=0)
    if 'LOFO' in C: o['LOFO'] = dl('LOFO', a, b)
    o['seeds'] = {s: dl(s, a, b) for s in SEEDS}; o['seeds0_4_pos'] = sum(o['seeds'][s] > 0 for s in SEEDS if int(s[4:]) < 5)
    o['seeds5_9_pos'] = sum(o['seeds'][s] > 0 for s in SEEDS if int(s[4:]) >= 5); o['seeds_mean'] = float(np.mean(list(o['seeds'].values())))
    R['pairs'][f'{a}-{b}'] = o; log(f'{a}-{b}', json.dumps({k: o.get(k) for k in ('fam10_exact', 'new34', 'LOFO', 'seeds_mean', 'seeds0_4_pos', 'seeds5_9_pos')}, default=float), o['boot'])
pf = R['pairs']['PF-P']; R['interaction'] = {k: (pf[k] - R['pairs']['AF-A'][k]) if isinstance(pf.get(k), float) else None for k in ('fam10_exact', 'LOFO', 'new34')}
R['interaction']['seeds'] = {s: pf['seeds'][s] - R['pairs']['AF-A']['seeds'][s] for s in SEEDS}
R['acceptance_PF-P'] = {'P1': [pf['fam10_exact'], pf['fam10_exact'] >= 0.002], 'P2': [pf['boot'].get('p_gt0'), (pf['boot'].get('p_gt0') or 0) >= 0.85],
    'P3': [pf.get('LOFO'), pf['seeds0_4_pos'], pf['seeds5_9_pos'], (pf.get('LOFO') or 0) > 0 and pf['seeds0_4_pos'] >= 4],
    'P4': [pf['per_heavy'], pf['new34'], all(v >= -0.0005 for v in pf['per_heavy'].values()) and pf['new34'] >= -0.001], 'P6': ['drop/drop', True]}
json.dump(R, open(W / 'pkg_gor.json', 'w'), ensure_ascii=False, indent=1, default=float)
import resource; log('done; peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
