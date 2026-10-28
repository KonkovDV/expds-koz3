"""bot-10: строгий П3 для реранкера архитектуры (копия логики /workspace/rerank/nested3.py).
LOFO по семьям; для внешней семьи k пороги подбираются на ВНУТРЕННЕМ 5-фолдовом OOF (случайные 5 групп семей обучения, seed=k),
а не на каноническом OOF. Только класс arkhitektury; прочие классы = база (их пороги не зависят от оценок arch).
Варианты признаков: full, noRG2 (без rg_n, rg_pct), noRGall (без rg_* и conf_rel).
Читает /workspace/rerank (только чтение), пишет только в bot-10/work4/rerank.
"""
import sys, json, pickle, time, argparse
sys.path.insert(0, '/workspace/rerank')
from common import *          # W, load, sets, family_folds, score_counts, paired_bootstrap_counts, site_family
import fast, numpy as np, lightgbm as lgb
from collections import Counter
from expds.metric import ClassCurve, curve_where, sweep_groups

OUT = Path('/workspace/koz3_shared/bot-10/work4/rerank')
ap = argparse.ArgumentParser(); ap.add_argument('--variants', default='full,noRG2,noRGall'); ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--layout', default='lofo'); ap.add_argument('--micro', default=''); ap.add_argument('--scope', default='global'); a = ap.parse_args()
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; N = len(cls); CL = fast.CL
X = np.load(W/'work/X2.npy', mmap_mode='r'); names = json.load(open(W/'work/X2_names.json')); pool = np.load(W/'work/pool.npy'); y = np.load(W/'work/first.npy')
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
C = fast.cid['arkhitektury']; A = 'arkhitektury'
arch = cls == C; ap_idx = np.flatnonzero(arch & pool)
Xa = np.asarray(X[ap_idx]); ya = y[ap_idx].astype(float); rega = reg[ap_idx]
print('arch candidates', int(arch.sum()), 'in pool', ap_idx.size, 'pos', int(ya.sum()), flush=True)
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
if a.layout.startswith('seed'):
    sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts'); from honest_bootstrap import random_family_folds
    folds = random_family_folds(regions, 10, int(a.layout[4:]))
else:
    folds = family_folds(regions, site_family, None if a.layout == 'lofo' else 10)
fold_of = {r: i for i, f in enumerate(folds) for r in f}
fid_a = np.array([fold_of[r] for r in rega]); fid_all_arch = np.array([fold_of[r] for r in reg[arch]])
fam_a = np.array([site_family(r) for r in rega])
PLS = {'orig': fast.PL[(A, A)]}
if a.micro:
    Pm, Jm, Sm = pickle.load(open(a.micro, 'rb')); d = {}
    o = np.lexsort((Jm, Pm))
    for p_, j_, s_ in zip(Pm[o], Jm[o], Sm[o]): d.setdefault(int(p_), []).append((int(j_), float(s_)))
    PLS['micro'] = d
if a.scope == 'site':  # сопоставление только внутри своего участка: пара (p, j) годна, если reg[p] == регион эталона j
    GR = T['greg'][A]
    PLS = {g: {p: [(j, s_) for j, s_ in l if GR[j] == T['reg'][p]] for p, l in d.items()} for g, d in PLS.items()}
    PLS = {g: {p: l for p, l in d.items() if l} for g, d in PLS.items()}
def arch_curve(score, geo='orig'):
    """fast.curves только для arkhitektury."""
    idx = np.flatnonzero(arch); o = idx[np.argsort(-score[idx], kind='stable')]
    hit = np.zeros(len(o), bool); matched = set()
    for r, p in enumerate(o):
        lst = PLS[geo].get(int(p))
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in lst:
            if j in matched: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: matched.add(bj); hit[r] = True
    return ClassCurve(conf=score[o].astype(np.float64), hit=hit, pred_region=list(T['reg'][o]), n_gt=T['n_gt'][A],
                      gt_regions=fast.GREG[A], pred_group=list(T['grp'][o]))
mono_all = [1 if n in ('conf', 'logit') else 0 for n in names]
FEATS = {'full': names, 'noRG2': [n for n in names if n not in ('rg_n', 'rg_pct')],
         'noRGall': [n for n in names if not n.startswith('rg_') and n != 'conf_rel']}
def train_pred(cols, tr, te):
    if ya[tr].sum() < 5 or not te.any(): return None
    ci = [names.index(n) for n in cols]
    P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, monotone_constraints=[mono_all[i] for i in ci], verbose=-1, num_threads=a.threads, seed=0)
    cat = [cols.index(n) for n in ('cls', 'grp', 'mk') if n in cols]
    return lgb.train(P, lgb.Dataset(Xa[tr][:, ci], ya[tr], categorical_feature=cat), 300).predict(Xa[te][:, ci])
# база: честные счётчики всех классов (pick=max, unseen=drop как в nested3)
def held_counts(cv_arch, taus, held, per):
    hc = curve_where(cv_arch, held)
    for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
    if hc.conf.size == 0: return
    tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
    for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
        per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
from expds.metric import honest_counts
if a.scope == 'site': base_cv[A] = arch_curve(conf.copy(), 'orig')
base_counts, _ = honest_counts(base_cv, folds)
res = {'layout': a.layout, 'folds': len(folds), 'base': score_counts(base_counts)[0]}
t0 = time.time()
eliz = {r for r in regions if site_family(r).startswith('Елизаветин')}; ex = regions - eliz
def summarize(name, per):
    cnt = dict(base_counts); cnt[A] = per
    sc = score_counts(cnt)[0]; bt = paired_bootstrap_counts(cnt, base_counts, site_family, 2000)
    pa, pb = score_counts(cnt)[1][A], score_counts(base_counts)[1][A]
    res[name] = dict(score=sc, d=sc - res['base'], boot=bt, new34=score_counts(cnt, new)[0] - score_counts(base_counts, new)[0],
                     d_excl_eliz=score_counts(cnt, ex)[0] - score_counts(base_counts, ex)[0],
                     arch_tp_fp=[pa['tp'], pa['fp']], base_arch_tp_fp=[pb['tp'], pb['fp']], arch_f1=[pb['f1'], pa['f1']])
    pickle.dump(cnt, open(OUT/f'counts_{a.layout}{"_site" if a.scope=="site" else ""}_{name}.pkl', 'wb'))
    print(name, f"d={sc-res['base']:+.6f} P={bt['p_gt0']:.3f} [{bt['p5']:+.5f};{bt['p95']:+.5f}] exclEliz={res[name]['d_excl_eliz']:+.6f} arch tp/fp {pb['tp']}/{pb['fp']}->{pa['tp']}/{pa['fp']} {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(OUT/f'res_{a.layout}_nested5{"_micro" if a.micro else ""}{"_site" if a.scope=="site" else ""}.json', 'w'), indent=1, default=float)
res['eliz_regions'] = sorted(eliz); res['feats'] = {k: len(v) for k, v in FEATS.items()}
for geo in PLS:  # контроль: без реранкера (исходные conf), пороги по обучающим фолдам
    per = {}; cv = arch_curve(conf.copy(), geo)
    for k, held in enumerate(folds):
        sw = sweep_groups({A: curve_where(cv, regions - held)})
        held_counts(cv, {g: gv['threshold'] for g, gv in sw[A]['groups'].items()}, held, per)
    summarize(f'conf_{geo}', per)
for vn in a.variants.split(','):
    cols = FEATS[vn]; pers = {geo: {} for geo in PLS}
    for k, held in enumerate(folds):
        if not (fid_all_arch == k).any():
            for geo in PLS:
                hc = curve_where(base_cv[A], held)
                for r, n in hc.gt_regions.items(): pers[geo].setdefault(r, [0, 0, 0])[2] += int(n)
            continue
        S = np.full(N, -1.0)
        trm = fid_a != k; te = fid_a == k
        p = train_pred(cols, trm, te)
        if p is not None: S[ap_idx[te]] = p
        tf = sorted(set(fam_a[trm])); rng = np.random.default_rng(k); lab = dict(zip(tf, rng.permutation(len(tf)) % 5))
        inner = np.array([lab.get(f, -1) for f in fam_a])
        for j in range(5):
            tr_j = trm & (inner != j); te_j = trm & (inner == j)
            q = train_pred(cols, tr_j, te_j)
            if q is not None: S[ap_idx[te_j]] = q
        for geo in PLS:
            cv = arch_curve(S, geo)
            sw = sweep_groups({A: curve_where(cv, regions - held)})
            held_counts(cv, {g: gv['threshold'] for g, gv in sw[A]['groups'].items()}, held, pers[geo])
    for geo in PLS: summarize(f'{vn}_{geo}', pers[geo])
pickle.dump(base_counts, open(OUT/f'counts_{a.layout}{"_site" if a.scope=="site" else ""}_base.pkl', 'wb'))
