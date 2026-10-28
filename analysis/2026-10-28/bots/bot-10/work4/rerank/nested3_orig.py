"""P3 check: nested leave-family-out reranker for selected classes on alternative fold layouts.
layout: 'canon' (family_folds 10, reuses nested_v2.npz), 'seedS' (random_family_folds as bot-10 honest_bootstrap), 'lofo'
(LOFO outer; thresholds on canon-10 OOF scores -> approximate)."""
from common import *
import fast, numpy as np, lightgbm as lgb, argparse, sys
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from expds.metric import curve_where, sweep_groups
ap = argparse.ArgumentParser(); ap.add_argument('--layout', default='seed0'); ap.add_argument('--classes', default='arkhitektury,finds_points')
ap.add_argument('--threads', type=int, default=3); a = ap.parse_args()
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; N = len(cls); CL = fast.CL
X = np.load(W/'work/X2.npy'); names = json.load(open(W/'work/X2_names.json')); pool = np.load(W/'work/pool.npy'); y = np.load(W/'work/first.npy')
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
def random_family_folds(regions, k, seed):
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    nm = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(nm)) % k
    return [set().union(*(fams[n] for n, l in zip(nm, lab) if l == i)) for i in range(k)]
canon = family_folds(regions, site_family, 10)
if a.layout == 'canon': folds = canon
elif a.layout == 'lofo': folds = family_folds(regions, site_family, None)
else: folds = random_family_folds(regions, 10, int(a.layout[4:]))
K = len(folds); fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in reg])
SEL = [fast.cid[c] for c in a.classes.split(',')]
mono = [1 if n in ('conf', 'logit') else 0 for n in names]; cat = [names.index(n) for n in ('cls', 'grp', 'mk')]
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, num_threads=a.threads, seed=0)
def tp(trainmask, pm):
    out = {}
    for c in SEL:
        tr = (cls == c) & trainmask & pool; m = pm & (cls == c) & pool
        if y[tr].sum() < 5 or not m.any(): continue
        out[c] = (m, lgb.train(P, lgb.Dataset(X[tr], y[tr].astype(float), categorical_feature=cat), 300).predict(X[m]))
    return out
def put(S, o):
    for c, (m, p) in o.items(): S[m] = p
cache = W/f'work/v3/nested_{a.layout}.npz'
if a.layout == 'canon':
    z = np.load(W/'work/nested_v2.npz'); outer = z['outer']
    def score_train(k): return z['inner'][k]
elif cache.exists():
    z = np.load(cache); outer = z['outer']; inner = z.get('inner')
    def score_train(k): return inner[k]
else:
    outer = np.full(N, -1.0); t = time.time()
    if a.layout == 'lofo':
        zc = np.load(W/'work/nested_v2.npz')['outer']  # canon OOF scores, used for train-side thresholds (approx.)
        for k in range(K): put(outer, tp(fid != k, fid == k))
        print('lofo trained', time.time() - t, flush=True); inner = None
        def score_train(k): return zc
        np.savez_compressed(cache, outer=outer)
    else:
        inner = np.full((K, N), -1.0)
        for k in range(K):
            put(outer, tp(fid != k, fid == k))
            for j in range(K):
                if j != k: put(inner[k], tp((fid != k) & (fid != j), fid == j))
            print('fold', k, f'{time.time()-t:.0f}s', flush=True)
        np.savez_compressed(cache, outer=outer, inner=inner)
        def score_train(k): return inner[k]
def mix(k, use):
    S = conf.copy(); trm = fid != k; st = score_train(k)
    for c in use:
        m = cls == c
        # rows outside pool keep -1 from the model arrays: put them below any pool score (same as nested2)
        S[m & trm] = st[m & trm]; S[m & ~trm] = outer[m & ~trm]
    return S
def honest_with(fn):
    counts = {c: {} for c in CL}
    for k, held in enumerate(folds):
        cv = fast.curves(score=fn(k)); train = regions - held
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
WTS = {'kurgany_povrezhdennye': 30, 'kurgany_tselye': 20, 'gorodishcha': 15, 'arkhitektury': 10, 'object_poly': 10, 'finds_points': 10, 'fortifikatsii': 5}
bc = honest_with(lambda k: conf); out = {'layout': a.layout, 'base': [score_counts(bc)[0], score_counts(bc, new)[0], score_counts(bc, optical)[0]]}
print(a.layout, 'base', out['base'], flush=True)
variants = {'arch': [fast.cid['arkhitektury']], 'finds': [fast.cid['finds_points']], 'arch+finds': SEL}
for nm, use in variants.items():
    if not set(use) <= set(SEL): continue
    cnt = honest_with(lambda k: mix(k, use)); s = [score_counts(cnt)[0], score_counts(cnt, new)[0], score_counts(cnt, optical)[0]]
    bt = paired_bootstrap_counts(cnt, bc, site_family, 1000); per, bper = score_counts(cnt)[1], score_counts(bc)[1]
    wd = {c: WTS[c] * (per[c]['f1'] - bper[c]['f1']) / 100 for c in CL if abs(per[c]['f1'] - bper[c]['f1']) > 0}
    out[nm] = dict(s=s, d=s[0] - out['base'][0], bt=bt, wdelta=wd)
    print(f"{a.layout} {nm:11s} all {s[0]:.6f} d {s[0]-out['base'][0]:+.5f} new {s[1]-out['base'][1]:+.4f} opt {s[2]-out['base'][2]:+.4f} | boot {bt['mean']:+.4f} [{bt['p5']:+.4f};{bt['p95']:+.4f}] P {bt['p_gt0']:.3f} | {wd}", flush=True)
    pickle.dump(cnt, open(W/f'work/v3/counts_{a.layout}_{nm}.pkl', 'wb'))
json.dump(out, open(W/f'work/v3/res_{a.layout}.json', 'w'), indent=1, default=float)
