"""[bot-10 work12: копия bot-6/work/orgate/gate_eval.py; OUT и feats -> bot-10/work12; добавлены assert вложенности]
bot-6 ORGATE: честная оценка гейта arch (логика копирует bot-10/work4/rerank/nested_lofo5.py, orig-сопоставление).
Внешний фолд = семьи; пороги — sweep_groups на внутреннем 5-фолдовом OOF обучающих семей. Прочие классы = база A.
Сравнение с A (база) и с R (bot-10 counts_<layout>_full_orig.pkl).
v2 (09:20): разбивка TP/FP arch по группам (or / sp_arch) для базы и вариантов; диагностика G_noLAS; ext: строки ok=False = NaN."""
import sys, json, pickle, time, argparse
sys.path.insert(0, '/workspace/rerank')
from common import *
import fast, numpy as np, lightgbm as lgb
from expds.metric import ClassCurve, curve_where, sweep_groups, honest_counts
OUT = Path('/workspace/koz3_shared/bot-10/work12/res'); OUT.mkdir(exist_ok=True)
B10 = Path('/workspace/koz3_shared/bot-10/work4/rerank')
ap = argparse.ArgumentParser(); ap.add_argument('--layout', default='k10'); ap.add_argument('--variants', default='Rrepro,RULE,G,G_noGeo,G_noRG')
ap.add_argument('--ext', default='', help='npz: stack_idx + (score OOF | emb -> PCA в фолде)'); ap.add_argument('--pca', type=int, default=16)
ap.add_argument('--threads', type=int, default=2); ap.add_argument('--scope', default='global'); a = ap.parse_args()
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; N = len(cls)
X2 = np.load(W/'work/X2.npy', mmap_mode='r'); n2 = json.load(open(W/'work/X2_names.json')); pool = np.load(W/'work/pool.npy'); y = np.load(W/'work/first.npy')
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
A = 'arkhitektury'; C = fast.cid[A]; arch = cls == C; ap_idx = np.flatnonzero(arch & pool)
fg = np.load('/workspace/koz3_shared/bot-10/work12/feats_gate.npz'); ng = list(fg['names'])
row = np.full(N, -1); row[fg['stack_idx']] = np.arange(len(fg['stack_idx'])); assert (row[ap_idx] >= 0).all()
XA = np.c_[np.asarray(X2[ap_idx]), fg['X'][row[ap_idx]]].astype(np.float32); names = n2 + ng
EXT = None
if a.ext:
    z = np.load(a.ext); r2 = np.full(N, -1); r2[z['stack_idx']] = np.arange(len(z['stack_idx'])); ri = r2[ap_idx]
    if 'score' in z: XA = np.c_[XA, np.where(ri >= 0, z['score'][np.maximum(ri, 0)], np.nan)]; names = names + ['ext_score']
    okz = z['ok'] if 'ok' in z else np.ones(len(z['stack_idx']), bool); has = (ri >= 0) & okz[np.maximum(ri, 0)]
    if 'emb' in z: EXT = np.where(has[:, None], z['emb'][np.maximum(ri, 0)].astype(np.float32), np.nan)
    print('ext coverage of arch pool', int(has.sum()), '/', len(ap_idx), flush=True)
ya = y[ap_idx].astype(float); rega = reg[ap_idx]
las = fg['X'][row[ap_idx], ng.index('r_las')] > 0; las_all = np.zeros(N, bool); las_all[fg['stack_idx']] = fg['X'][:, ng.index('r_las')] > 0
print('arch', int(arch.sum()), 'pool', ap_idx.size, 'pos', int(ya.sum()), 'feats', len(names), flush=True)
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
if a.layout.startswith('seed'):
    sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts'); from honest_bootstrap import random_family_folds
    folds = random_family_folds(regions, 10, int(a.layout[4:]))
else: folds = family_folds(regions, site_family, None if a.layout == 'lofo' else 10)
fold_of = {r: i for i, f in enumerate(folds) for r in f}
fid_a = np.array([fold_of[r] for r in rega]); fid_all_arch = np.array([fold_of[r] for r in reg[arch]]); fam_a = np.array([site_family(r) for r in rega])
PL = fast.PL[(A, A)]
if a.scope == 'site':
    GR = T['greg'][A]; PL = {p: [(j, s_) for j, s_ in l if GR[j] == T['reg'][p]] for p, l in PL.items()}; PL = {p: l for p, l in PL.items() if l}
def arch_curve(score):
    idx = np.flatnonzero(arch); o = idx[np.argsort(-score[idx], kind='stable')]; hit = np.zeros(len(o), bool); matched = set()
    for r, p in enumerate(o):
        lst = PL.get(int(p))
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in lst:
            if j in matched: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: matched.add(bj); hit[r] = True
    return ClassCurve(conf=score[o].astype(np.float64), hit=hit, pred_region=list(T['reg'][o]), n_gt=T['n_gt'][A], gt_regions=fast.GREG[A], pred_group=list(T['grp'][o]))
GEO = {'lat', 'lon'}; RG = {n for n in n2 if n.startswith('rg_')} | {'conf_rel'}
FEATS = {'Rrepro': n2, 'G': names, 'G_noLAS': names, 'G_noGeo': [n for n in names if n not in GEO], 'G_noRG': [n for n in names if n not in GEO | RG]}
if a.ext: FEATS['Gext'] = names; FEATS['G'] = [n for n in names if n != 'ext_score']; FEATS['G_noGeo'] = [n for n in FEATS['G'] if n not in GEO]
RULE_ON = {'Rrepro': False, 'G_noLAS': False}
def train_pred(cols, tr, te, use_emb):
    if ya[tr].sum() < 5 or not te.any(): return None
    ci = [names.index(n) for n in cols]; Xtr, Xte = XA[tr][:, ci], XA[te][:, ci]; mono = [1 if n in ('conf', 'logit') else 0 for n in cols]
    if use_emb and EXT is not None:  # PCA только на обучающих строках фолда
        from sklearn.decomposition import PCA
        ok = ~np.isnan(EXT[tr][:, 0]); pca = PCA(a.pca, random_state=0).fit(EXT[tr][ok][:20000])
        f = lambda M: np.where(np.isnan(M[:, :1]), np.nan, pca.transform(np.nan_to_num(M)))
        Xtr, Xte = np.c_[Xtr, f(EXT[tr])], np.c_[Xte, f(EXT[te])]; mono += [0] * a.pca
    P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, num_threads=a.threads, seed=0)
    cat = [cols.index(n) for n in ('cls', 'grp', 'mk') if n in cols]
    return lgb.train(P, lgb.Dataset(Xtr, ya[tr], categorical_feature=cat), 300).predict(Xte)
def held_counts(cv_arch, taus, held, per, pg=None):
    hc = curve_where(cv_arch, held)
    for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
    if hc.conf.size == 0: return
    tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
    for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    if pg is not None:
        for g, h in zip(np.asarray(hc.pred_group, dtype=object)[kept], hc.hit[kept]): pg.setdefault(str(g), [0, 0])[0 if h else 1] += 1
if a.scope == 'site': base_cv[A] = arch_curve(conf.copy())
base_counts, _ = honest_counts(base_cv, folds)
SS = '_site' if a.scope == 'site' else ''
Rp = B10/f'counts_{a.layout}{SS}_full_orig.pkl'; Rc = pickle.load(open(Rp, 'rb')) if Rp.exists() else None
PGB = {}
for k, held in enumerate(folds):
    _sw = sweep_groups({A: curve_where(base_cv[A], regions - held)}); held_counts(base_cv[A], {g: gv['threshold'] for g, gv in _sw[A]['groups'].items()}, held, {}, PGB)
res = {'base_grp_tp_fp': PGB, 'layout': a.layout, 'folds': len(folds), 'base': score_counts(base_counts)[0], 'R_file': str(Rp) if Rc else None}
if Rc: res['R'] = score_counts(Rc)[0]
eliz = {r for r in regions if site_family(r).startswith('Елизаветин')}; ex = regions - eliz; t0 = time.time()
WTS = {'kurgany_povrezhdennye': 30, 'kurgany_tselye': 20, 'gorodishcha': 15, 'arkhitektury': 10, 'object_poly': 10, 'finds_points': 10, 'fortifikatsii': 5}
def summarize(name, per, pg=None):
    cnt = dict(base_counts); cnt[A] = per; sc, pc = score_counts(cnt); bt = paired_bootstrap_counts(cnt, base_counts, site_family, 2000)
    pa, pb = pc[A], score_counts(base_counts)[1][A]
    d = dict(score=sc, d=sc - res['base'], boot=bt, new34=score_counts(cnt, new)[0] - score_counts(base_counts, new)[0],
             optical=score_counts(cnt, optical)[0] - score_counts(base_counts, optical)[0],
             d_excl_eliz=score_counts(cnt, ex)[0] - score_counts(base_counts, ex)[0],
             arch_tp_fp=[pa['tp'], pa['fp']], base_arch_tp_fp=[pb['tp'], pb['fp']], arch_f1=[pb['f1'], pa['f1']], wdelta_arch=WTS[A] * (pa['f1'] - pb['f1']) / 100)
    if Rc:
        bR = paired_bootstrap_counts(cnt, Rc, site_family, 2000)
        d.update(dR=sc - res['R'], bootR=bR, dR_excl_eliz=d['d_excl_eliz'] - (score_counts(Rc, ex)[0] - score_counts(base_counts, ex)[0]),
                 R_arch_tp_fp=[score_counts(Rc)[1][A]['tp'], score_counts(Rc)[1][A]['fp']])
    if pg is not None: d['grp_tp_fp'] = pg
    res[name] = d; pickle.dump(cnt, open(OUT/f'counts_{a.layout}{SS}_{name}.pkl', 'wb'))
    print(f"{a.layout} {name:8s} dA={d['d']:+.6f} P={bt['p_gt0']:.3f} [{bt['p5']:+.5f};{bt['p95']:+.5f}] exE={d['d_excl_eliz']:+.6f} new34={d['new34']:+.5f} "
          f"arch {pb['tp']}/{pb['fp']}->{pa['tp']}/{pa['fp']}" + (f" | dR={d['dR']:+.6f} P_R={d['bootR']['p_gt0']:.3f} dR_exE={d['dR_excl_eliz']:+.6f} R {d['R_arch_tp_fp']}" if Rc else '') + f" {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(OUT/f'res_{a.layout}{SS}{"_ext" if a.ext else ""}.json', 'w'), indent=1, default=float)
for vn in a.variants.split(','):
    if vn not in FEATS and vn != 'RULE': continue
    per = {}; pg = {}
    if vn == 'RULE':
        S = conf.copy(); S[las_all & arch] = -1; cv = arch_curve(S)
        for k, held in enumerate(folds):
            sw = sweep_groups({A: curve_where(cv, regions - held)}); held_counts(cv, {g: gv['threshold'] for g, gv in sw[A]['groups'].items()}, held, per, pg)
        summarize(vn, per, pg); continue
    cols = FEATS[vn]; rule = RULE_ON.get(vn, True); ue = vn == 'Gext'
    for k, held in enumerate(folds):
        if not (fid_all_arch == k).any():
            hc = curve_where(base_cv[A], held)
            for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            continue
        S = np.full(N, -1.0); trm = fid_a != k; te = fid_a == k
        assert not set(fam_a[trm]) & set(fam_a[te]), 'семья в обучении и тесте'
        assert not set(rega[trm]) & held and set(rega[te]) <= held
        p = train_pred(cols, trm, te, ue)
        if p is not None: S[ap_idx[te]] = p
        tf = sorted(set(fam_a[trm])); rng = np.random.default_rng(k); lab = dict(zip(tf, rng.permutation(len(tf)) % 5)); inner = np.array([lab.get(f, -1) for f in fam_a])
        assert (inner[te] == -1).all() and (inner[trm] >= 0).all(), 'внешний фолд во внутреннем CV'
        for j in range(5):
            q = train_pred(cols, trm & (inner != j), trm & (inner == j), ue)
            if q is not None: S[ap_idx[trm & (inner == j)]] = q
        if rule: S[las_all & arch] = -1
        assert not (regions - held) & held
        cv = arch_curve(S); sw = sweep_groups({A: curve_where(cv, regions - held)})
        held_counts(cv, {g: gv['threshold'] for g, gv in sw[A]['groups'].items()}, held, per, pg)
    summarize(vn, per, pg)
pickle.dump(base_counts, open(OUT/f'counts_{a.layout}{SS}_base.pkl', 'wb'))
