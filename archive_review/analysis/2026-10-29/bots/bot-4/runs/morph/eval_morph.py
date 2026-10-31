"""bot-4 MORPH: строгий nested-реранкер лидарных курганов по PREREG_MORPH.md (29.10 08:43 МСК).
Обобщение bot-4/work/cropverif/evalrr.py и bot-10/work4/rerank/nested_lofo5.py на li-курганы (не оптический пул).
Варианты: M = conf,logit,cls,grp,mk + 45 morph (morph_M.npz, фиксированный список PREREG); C = только conf,logit,cls,grp,mk (атрибуция).
Строки реранжирования R: li_kurgan_candidates.npz (mk li, cls povr/tselye) с morph_valid >= 0.05 и без NaN в morph_valid.
Остальные строки стека (в т.ч. li-курганы с morph_valid<0.05) — исходный conf (не -1). Unseen=keep.
Один LightGBM на оба класса курганов (cls — категориальный признак). Метка work/first.npy.
Пороги: для внешнего фолда k — внутренний 5-фолдовый OOF по семьям обучения (seed=k), sweep_groups на обучающих участках.
Бины LightGBM строятся один раз по признакам всех строк R (без меток), обучение — Dataset.subset (ускорение; меток не использует).
Пишет только в bot-4/work/morph/res/."""
import sys, json, pickle, time, argparse, os, resource, gc
sys.path.insert(0, '/workspace/rerank')
from common import *
import fast, numpy as np, lightgbm as lgb
from expds.metric import curve_where, sweep_groups
import expds.metric as M
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from honest_bootstrap import random_family_folds, weighted_class_delta
ap = argparse.ArgumentParser(); ap.add_argument('--layouts', default='k10'); ap.add_argument('--variants', default='C,M')
ap.add_argument('--maxfolds', type=int, default=0); ap.add_argument('--threads', type=int, default=2); ap.add_argument('--tag', default='')
ap.add_argument('--basecheck', action='store_true'); ap.add_argument('--nboot', type=int, default=2000)
a = ap.parse_args()
D = Path('/workspace/koz3_shared/bot-4/work/morph'); OUT = D/'res'; OUT.mkdir(exist_ok=True)
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; grp = np.asarray(T['grp']); N = len(cls)
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
def layout(ln):
    if ln.startswith('seed'): return random_family_folds(regions, 10, int(ln[4:]))
    return M.family_folds(regions, site_family, None if ln == 'lofo' else 10)
if a.basecheck:
    for pol in ('drop', 'keep'):
        c, info = M.honest_counts(base_cv, layout('k10'), unseen=pol); print('basecheck', pol, score_counts(c)[0], info, flush=True)
    sys.exit()
X2 = np.load(W/'work/X2.npy', mmap_mode='r'); x2n = json.load(open(W/'work/X2_names.json')); y = np.load(W/'work/first.npy')
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
BASE = ['conf', 'logit', 'cls', 'grp', 'mk']; assert x2n[:5] == BASE
mz = np.load(D/'morph_M.npz', allow_pickle=False); mnames = [str(s) for s in mz['names']]
PRE = ['morph_valid', 'morph_bg_slope', 'morph_bg_rough', 'morph_h0', 'morph_hmax', 'morph_r_peak', 'morph_ring', 'morph_ring_rel', 'morph_r_half', 'morph_r_edge',
       'morph_slope_edge', 'morph_vol', 'morph_shape', 'morph_top_ratio', 'morph_asym4', 'morph_secmin4', 'morph_secmin8', 'morph_elong', 'morph_cent_off', 'morph_peak_off',
       'morph_tpi2', 'morph_tpi5', 'morph_tpi8', 'morph_std3', 'morph_std8', 'morph_slope2', 'morph_slope5', 'morph_slope_max5', 'morph_curv1_5', 'morph_curv3',
       'morph_rz_bg_rough', 'morph_rz_h0', 'morph_rz_hmax', 'morph_rz_ring', 'morph_rz_r_half', 'morph_rz_r_edge', 'morph_rz_slope_edge', 'morph_rz_vol', 'morph_rz_tpi2',
       'morph_rz_tpi5', 'morph_rz_tpi8', 'morph_rz_std3', 'morph_rz_std8', 'morph_rz_curv1_5', 'morph_rz_curv3']
assert mnames == PRE, 'список признаков не совпадает с PREREG'
lk = np.load('/workspace/koz3_shared/bot-4/work/cropverif/li_kurgan_candidates.npz', allow_pickle=False)
assert (lk['stack_idx'] == mz['stack_idx']).all()
sidx = mz['stack_idx'].astype(np.int64); mv = mz['X'][:, 0]
assert np.isin(cls[sidx], [0, 1]).all() and (T['mk'][sidx] == 'li').all()
ok = np.isfinite(mv) & (mv >= 0.05)
R = sidx[ok]; o = np.argsort(R); R = R[o]
MX = mz['X'][ok][o]; del mz
yR = y[R].astype(float); regR = reg[R]
B5 = np.asarray(X2[R][:, :5], dtype=np.float32)
assert np.allclose(B5[:, 0], conf[R], atol=1e-6) and (B5[:, 2] == cls[R]).all()
print('li kurgan rows', sidx.size, 'rerank R', R.size, 'pos', int(yR.sum()), 'left (valid<0.05)', int((~ok).sum()), 'pos left', int(y[sidx[~ok]].sum()), flush=True)
FEATS = {'C': (B5, BASE), 'M': (np.hstack([B5, MX]), BASE + PRE)}
del MX
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=a.threads, seed=0)
DS = {}
for v in a.variants.split(','):
    F, nm = FEATS[v]
    p = dict(P, monotone_constraints=[1 if n in ('conf', 'logit') else 0 for n in nm])
    ds = lgb.Dataset(F, yR, feature_name=nm, categorical_feature=['cls', 'grp', 'mk'], params=dict(p), free_raw_data=False).construct()
    DS[v] = (ds, F, p)
for v in list(FEATS):
    if v not in DS: del FEATS[v]
def fit_pred(v, tr, te):
    if yR[tr].sum() < 5 or not te.any(): return None
    ds, F, p = DS[v]
    b = lgb.train(p, ds.subset(np.flatnonzero(tr)), 300)
    return b.predict(F[te])
def held_counts(cv, held, cnt):
    res = sweep_groups({c: curve_where(cv[c], regions - held) for c in cv})
    taus = {}
    for c, vv in res.items():
        for g, gv in vv['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
    for c, cc in cv.items():
        hc = curve_where(cc, held); per = cnt.setdefault(c, {})
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        if hc.conf.size == 0: continue
        tau = np.array([(-np.inf if taus.get(g, {}).get(c) is None else taus[g][c]) for g in hc.pred_group])  # unseen=keep
        kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
famname = np.array([site_family(r) for r in regR])
LAY = {}
for ln in a.layouts.split(','):
    folds = layout(ln); bc, info = M.honest_counts(base_cv, folds, unseen='keep'); LAY[ln] = (folds, bc, info)
del base_cv; gc.collect()
fams = {}
for r in regions: fams.setdefault(site_family(r), set()).add(r)
for ln in a.layouts.split(','):
    folds, base_counts, info = LAY[ln]; fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in regR])
    bsc = score_counts(base_counts)[0]
    print(ln, 'folds', len(folds), 'base keep', bsc, info, flush=True)
    cnts = {v: {} for v in DS}; t0 = time.time()
    for k, held in enumerate(folds):
        if a.maxfolds and k >= a.maxfolds: break
        trm = fid != k; te = fid == k
        if not te.any():
            bcv = fast.curves()
            for v in cnts: held_counts(bcv, held, cnts[v])
            del bcv; gc.collect()
            continue
        tf = sorted(set(famname[trm])); rng = np.random.default_rng(k); lab = dict(zip(tf, rng.permutation(len(tf)) % 5))
        inner = np.array([lab.get(f, -1) for f in famname])
        for v in cnts:
            S = conf.astype(np.float64).copy(); filled = np.zeros(R.size, bool)
            p = fit_pred(v, trm, te)
            if p is not None: S[R[te]] = p; filled |= te
            for j in range(5):
                m_ = trm & (inner == j); q = fit_pred(v, trm & (inner != j), m_)
                if q is not None: S[R[m_]] = q; filled |= m_
            if not filled.all(): print(f'  WARN {ln} fold {k} {v}: unfilled R rows keep conf: {int((~filled).sum())}', flush=True)
            cv_ = fast.curves(score=S); del S; held_counts(cv_, held, cnts[v]); del cv_; gc.collect()
        print(f'  {ln} fold {k+1}/{len(folds)} {time.time()-t0:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}MB', flush=True)
    res = {'layout': ln, 'folds': len(folds), 'base': bsc, 'base_info': info, 'maxfolds': a.maxfolds, 'R': int(R.size), 'R_pos': int(yR.sum())}
    if a.maxfolds:
        sub = set().union(*folds[:a.maxfolds]); res['partial_regions'] = len(sub)
        for v, c in cnts.items(): res[v] = score_counts(c, sub)[0] - score_counts(base_counts, sub)[0]
        print(json.dumps(res, default=float), flush=True); continue
    for v, c in cnts.items():
        sc = score_counts(c)[0]; bt = paired_bootstrap_counts(c, base_counts, site_family, a.nboot)
        wd = weighted_class_delta(c, base_counts)
        lofam = sorted(((fn, (sc - score_counts(c, regions - fs)[0]) - (bsc - score_counts(base_counts, regions - fs)[0])) for fn, fs in fams.items()), key=lambda t: -abs(t[1]))[:6]
        res[v] = dict(score=sc, d=sc - bsc, boot=bt, new34=score_counts(c, new)[0] - score_counts(base_counts, new)[0],
                      optical=score_counts(c, optical)[0] - score_counts(base_counts, optical)[0],
                      weighted={k_: round(x['weighted'], 6) for k_, x in wd.items()}, family_contrib_top6=lofam)
        pickle.dump(c, open(OUT/f'counts_{ln}_{v}{a.tag}.pkl', 'wb'))
        print(ln, v, f"d={sc-bsc:+.6f} P={bt['p_gt0']:.3f} [{bt['p5']:+.5f};{bt['p95']:+.5f}] new34={res[v]['new34']:+.6f} opt={res[v]['optical']:+.6f}", res[v]['weighted'], flush=True)
    if 'M' in cnts and 'C' in cnts:
        res['M_minus_C'] = dict(d=res['M']['score'] - res['C']['score'], boot=paired_bootstrap_counts(cnts['M'], cnts['C'], site_family, a.nboot))
        print(ln, 'M-C', res['M_minus_C'], flush=True)
    res['maxrss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024; res['sec'] = time.time() - t0; print('maxrss', res['maxrss_mb'], flush=True)
    pickle.dump(base_counts, open(OUT/f'counts_{ln}_base.pkl', 'wb'))
    json.dump(res, open(OUT/f'res_{ln}{a.tag}.json', 'w'), indent=1, default=float, ensure_ascii=False)
