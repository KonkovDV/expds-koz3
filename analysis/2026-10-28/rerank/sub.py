"""Nested honest damaged-vs-intact subclass classifier for kurgan candidates.
p = P(povr | kurgan hit). Variants: rescore S = conf * p_own^a ; drop twin copy with p_own < d."""
from common import *
import fast, numpy as np, lightgbm as lgb
from expds.metric import curve_where, sweep_groups
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; N = len(cls); CL = fast.CL
X = np.load(W/'work/X2.npy'); names = json.load(open(W/'work/X2_names.json')); okc = np.load(W/'work/okc.npy')
twin = np.load(W/'work/twin.npy')
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10); K = len(folds)
fold_of = {r: i for i, f in enumerate(folds) for r in f}
fid = np.array([fold_of[r] for r in reg])
KM = (cls == 0) | (cls == 1)
lab = KM & (okc[:, 0] ^ okc[:, 1]); y = okc[:, 0]
cat = [names.index(n) for n in ('cls', 'grp', 'mk')]
P = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=0)
def tp(train, pm):
    tr = lab & np.isin(fid, list(train))
    m = lgb.train(P, lgb.Dataset(X[tr], y[tr].astype(float), categorical_feature=cat), 200)
    return m.predict(X[pm])
outer = np.full(N, 0.5); inner = np.full((K, N), 0.5)
for k in range(K):
    t = time.time(); pm = KM & (fid == k); outer[pm] = tp(set(range(K)) - {k}, pm)
    for j in range(K):
        if j == k: continue
        pm = KM & (fid == j); inner[k][pm] = tp(set(range(K)) - {k, j}, pm)
    print('fold', k, f'{time.time()-t:.0f}s', flush=True)
np.savez_compressed(W/'work/sub.npz', outer=outer, inner=inner)
from sklearn.metrics import roc_auc_score
print('OOF AUC povr-vs-tsel on labelled', roc_auc_score(y[lab], outer[lab]), 'base (cls==povr) acc', ((cls[lab] == 0) == y[lab]).mean(),
      'model acc', ((outer[lab] > .5) == y[lab]).mean(), flush=True)
def pown(p): return np.where(cls == 0, p, 1 - p)
def honest_fn(make):  # make(p) -> (score, keep)
    counts = {c: {} for c in CL}
    for k, held in enumerate(folds):
        tr = fid != k; p = np.where(tr, inner[k], outer)
        S, keep = make(p)
        cv = fast.curves(score=S, keep=keep); train = regions - held
        res = sweep_groups({c: curve_where(v, train) for c, v in cv.items()})
        taus = {}
        for c, v in res.items():
            for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
        for c, v in cv.items():
            hc = curve_where(v, held); per = counts[c]
            for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0: continue
            tau = np.array([taus.get(g, {}).get(c, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return counts
bc = honest_fn(lambda p: (conf, None)); bper = score_counts(bc)[1]
print('base', score_counts(bc)[0], score_counts(bc, new)[0], score_counts(bc, optical)[0])
res = {}
def rep(name, cnt):
    s = [score_counts(cnt)[0], score_counts(cnt, new)[0], score_counts(cnt, optical)[0]]; per = score_counts(cnt)[1]
    bt = paired_bootstrap_counts(cnt, bc, site_family, 1000)
    print(f"{name:28s} all {s[0]:.6f} new {s[1]:.6f} opt {s[2]:.6f} | d {bt['mean']:+.4f} [{bt['p5']:+.4f};{bt['p95']:+.4f}] P {bt['p_gt0']:.3f} | povr {per[CL[0]]['f1']-bper[CL[0]]['f1']:+.4f} tsel {per[CL[1]]['f1']-bper[CL[1]]['f1']:+.4f}", flush=True)
    res[name] = dict(s=s, bt=bt); pickle.dump(cnt, open(W/f'work/counts_sub_{name}.pkl', 'wb'))
for a in (0.5, 1.0, 2.0):
    rep(f'rescore a={a}', honest_fn(lambda p, a=a: (np.where(KM, conf * pown(p) ** a, conf), None)))
for d in (0.2, 0.3, 0.4, 0.5):
    rep(f'drop twin p_own<{d}', honest_fn(lambda p, d=d: (conf, ~(KM & (twin >= 0) & (pown(p) < d)))))
json.dump(res, open(W/'work/sub.json', 'w'), indent=1, default=float)
