"""bot-4 CROPVERIF: строгий nested-реранкер по PREREG_CROPVERIF.md (обобщение bot-10/work4/rerank/nested_lofo5.py на 3 семейства классов, unseen=keep).
Варианты: X = X2 (35 призн.), E = X2 + PCA32(DINOv2-B/reg). Пишет только в bot-4/work/cropverif/res/."""
import sys, json, pickle, time, argparse, os
sys.path.insert(0, '/workspace/rerank')
from common import *
import fast, numpy as np, lightgbm as lgb
from expds.metric import curve_where, sweep_groups
import expds.metric as M
ap = argparse.ArgumentParser(); ap.add_argument('--layouts', default='k10,lofo,seed0,seed1,seed2,seed3,seed4')
ap.add_argument('--variants', default='X,E'); ap.add_argument('--maxfolds', type=int, default=0); ap.add_argument('--emb', default='emb_dinov2b_reg_112.npz')
ap.add_argument('--npc', type=int, default=32); ap.add_argument('--threads', type=int, default=2); ap.add_argument('--tag', default='')
a = ap.parse_args()
OUT = Path('/workspace/koz3_shared/bot-4/work/cropverif/res'); OUT.mkdir(exist_ok=True)
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; grp = np.asarray(T['grp']); N = len(cls); CL = T['CL']
X2 = np.load(W/'work/X2.npy', mmap_mode='r'); names = json.load(open(W/'work/X2_names.json')); pool = np.load(W/'work/pool.npy'); y = np.load(W/'work/first.npy')
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
FAM = {0: 'kur', 1: 'kur', 3: 'arch', 5: 'fort'}
FAMC = np.array([FAM.get(i, '-') for i in range(7)])
oc = np.load('optical_pool_cands.npz', allow_pickle=False)
R = oc['stack_idx'].astype(np.int64); R = R[np.isin(cls[R], list(FAM))]; R.sort()
assert pool[R].all()
famR = np.array([FAM[c] for c in cls[R]]); yR = y[R].astype(float); regR = reg[R]
XR = np.asarray(X2[R], dtype=np.float32)
ez = np.load(a.emb, allow_pickle=False); pos = {int(s): i for i, s in enumerate(ez['stack_idx'])}
ei = np.array([pos.get(int(s), -1) for s in R]); okR = (ei >= 0) & ez['ok'][np.maximum(ei, 0)]
ER = np.full((len(R), ez['emb'].shape[1]), np.nan, np.float32); ER[okR] = ez['emb'][ei[okR]].astype(np.float32)
print('rows', len(R), {f: int((famR == f).sum()) for f in ('kur', 'arch', 'fort')}, 'pos', {f: int(yR[famR == f].sum()) for f in ('kur', 'arch', 'fort')}, 'emb ok', int(okR.sum()), flush=True)
# группы, которые реранжируются: вне пула в них -> -1 (как nested_lofo5)
rr_cls = np.isin(cls, list(FAM)); rgrp = set(zip(grp[R].tolist(), cls[R].tolist()))
nonpool_rr = np.flatnonzero(rr_cls & ~pool & np.array([(g, c) in rgrp for g, c in zip(grp, cls)]))
nonpool_rr = np.setdiff1d(nonpool_rr, R)
print('вне пула в реранжируемых группах -> -1:', nonpool_rr.size, 'групп', len(rgrp), flush=True)
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
mono = [1 if n in ('conf', 'logit') else 0 for n in names]
def fit_pred(F, mon, tr, te):
    if yR[tr].sum() < 5 or not te.any(): return None
    P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, monotone_constraints=mon, verbose=-1, num_threads=a.threads, seed=0)
    cat = [names.index(n) for n in ('cls', 'grp', 'mk')]
    return lgb.train(P, lgb.Dataset(F[tr], yR[tr], categorical_feature=cat), 300).predict(F[te])
def pca(tr):
    m = tr & okR; mu = ER[m].mean(0, dtype=np.float64); C = np.zeros((ER.shape[1],) * 2)
    Em = ER[m]
    for s_ in range(0, len(Em), 4096):
        B = Em[s_:s_ + 4096].astype(np.float64) - mu; C += B.T @ B
    w, Vv = np.linalg.eigh(C); V = Vv[:, ::-1][:, :a.npc]
    Z = np.full((len(R), a.npc), np.nan, np.float32); Z[okR] = ((ER[okR] - mu) @ V).astype(np.float32); return Z
def held_counts(cv, held, cnt):
    res = sweep_groups({c: curve_where(cv[c], regions - held) for c in cv})
    taus = {}
    for c, v in res.items():
        for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
    for c, cc in cv.items():
        hc = curve_where(cc, held); per = cnt.setdefault(c, {})
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        if hc.conf.size == 0: continue
        tau = np.array([(-np.inf if taus.get(g, {}).get(c) is None else taus[g][c]) for g in hc.pred_group])  # unseen=keep
        kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
def layout(ln):
    if ln.startswith('seed'):
        sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts'); from honest_bootstrap import random_family_folds
        return random_family_folds(regions, 10, int(ln[4:]))
    return M.family_folds(regions, site_family, None if ln == 'lofo' else 10)
famname = np.array([site_family(r) for r in regR])
for ln in a.layouts.split(','):
    folds = layout(ln); fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in regR])
    base_counts, info = M.honest_counts(base_cv, folds, unseen='keep'); bsc = score_counts(base_counts)[0]
    print(ln, 'folds', len(folds), 'base', bsc, info, flush=True)
    cnts = {v: {} for v in a.variants.split(',')}; t0 = time.time()
    for k, held in enumerate(folds):
        if a.maxfolds and k >= a.maxfolds: break
        trm = fid != k; te = fid == k
        if not te.any():
            for v in cnts: held_counts(base_cv, held, cnts[v])
            continue
        tf = sorted(set(famname[trm])); rng = np.random.default_rng(k); lab = dict(zip(tf, rng.permutation(len(tf)) % 5))
        inner = np.array([lab.get(f, -1) for f in famname])
        Z = pca(trm) if any(v.startswith('E') for v in cnts) else None
        for v in cnts:
            F = XR if v.startswith('X') else np.hstack([XR, Z]); mon = mono + [0] * (F.shape[1] - len(names))
            vf = v.split(':')[1].split('+') if ':' in v else ['kur', 'arch', 'fort']
            S = conf.astype(np.float64).copy(); rrm = np.isin(famR, vf)
            S[nonpool_rr[np.isin(FAMC[cls[nonpool_rr]], vf)]] = -1.0; S[R[rrm]] = -1.0
            for f in vf:
                fm = famR == f
                p = fit_pred(F, mon, trm & fm, te & fm)
                if p is not None: S[R[te & fm]] = p
                for j in range(5):
                    q = fit_pred(F, mon, trm & fm & (inner != j), trm & fm & (inner == j))
                    if q is not None: S[R[trm & fm & (inner == j)]] = q
            held_counts(fast.curves(score=S), held, cnts[v])
        print(f'  {ln} fold {k+1}/{len(folds)} {time.time()-t0:.0f}s', flush=True)
    res = {'layout': ln, 'folds': len(folds), 'base': bsc, 'base_info': info, 'maxfolds': a.maxfolds}
    if a.maxfolds:
        sub = set().union(*folds[:a.maxfolds]); res['partial_regions'] = len(sub)
        for v, c in cnts.items(): res[v] = score_counts(c, sub)[0] - score_counts(base_counts, sub)[0]
        print(json.dumps(res, default=float), flush=True); continue
    sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts'); from honest_bootstrap import weighted_class_delta
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    for v, c in cnts.items():
        sc = score_counts(c)[0]; bt = paired_bootstrap_counts(c, base_counts, site_family, 2000)
        wd = weighted_class_delta(c, base_counts)
        lofam = sorted(((fn, (sc - score_counts(c, regions - fs)[0]) - (bsc - score_counts(base_counts, regions - fs)[0])) for fn, fs in fams.items()), key=lambda t: -abs(t[1]))[:6]
        res[v] = dict(score=sc, d=sc - bsc, boot=bt, new34=score_counts(c, new)[0] - score_counts(base_counts, new)[0],
                      optical=score_counts(c, optical)[0] - score_counts(base_counts, optical)[0],
                      weighted={k_: round(x['weighted'], 6) for k_, x in wd.items()}, family_contrib_top6=lofam)
        pickle.dump(c, open(OUT/f'counts_{ln}_{v.replace(":", "_").replace("+", "")}{a.tag}.pkl', 'wb'))
        print(ln, v, f"d={sc-bsc:+.6f} P={bt['p_gt0']:.3f} [{bt['p5']:+.5f};{bt['p95']:+.5f}] new34={res[v]['new34']:+.6f} opt={res[v]['optical']:+.6f}", res[v]['weighted'], flush=True)
    if 'E' in cnts and 'X' in cnts:
        res['E_minus_X'] = dict(d=res['E']['score'] - res['X']['score'], boot=paired_bootstrap_counts(cnts['E'], cnts['X'], site_family, 2000))
        print(ln, 'E-X', res['E_minus_X'], flush=True)
    for v in cnts:
        if v.startswith('E:') and 'X:' + v[2:] in cnts:
            res['E_minus_X:' + v[2:]] = dict(d=res[v]['score'] - res['X:' + v[2:]]['score'], boot=paired_bootstrap_counts(cnts[v], cnts['X:' + v[2:]], site_family, 2000))
            print(ln, 'E-X', v[2:], res['E_minus_X:' + v[2:]], flush=True)
    import resource; res['maxrss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024; print('maxrss', res['maxrss_mb'], flush=True)
    pickle.dump(base_counts, open(OUT/f'counts_{ln}_base.pkl', 'wb'))
    json.dump(res, open(OUT/f'res_{ln}{a.tag}.json', 'w'), indent=1, default=float, ensure_ascii=False)
