"""Nested leave-family-out reranker: outer fold k -> model on other 9 folds; thresholds for k swept on
inner OOF scores of the other 9 folds (models that saw neither k nor the scored fold)."""
from common import *
import fast, numpy as np, lightgbm as lgb, os, argparse
from expds.metric import curve_where, sweep_groups
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v1'); ap.add_argument('--rounds', type=int, default=300)
ap.add_argument('--neg', type=float, default=0.2); ap.add_argument('--drop', default=''); ap.add_argument('--threads', type=int, default=4); ap.add_argument('--label', default='first'); ap.add_argument('--pergroup', action='store_true'); ap.add_argument('--xset', default='X2')
args = ap.parse_args()
T = fast.T; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; N = len(cls); CL = fast.CL
X = np.load(W/f'work/{args.xset}.npy'); names = json.load(open(W/f'work/{args.xset}_names.json')); okc = np.load(W/'work/okc.npy'); pool = np.load(W/'work/pool.npy')
keepcols = [i for i, n in enumerate(names) if n not in args.drop.split(',')]
X = X[:, keepcols]; names = [names[i] for i in keepcols]
y = okc[np.arange(N), np.maximum(cls, 0)] & (cls >= 0)
if args.label == 'first': y = np.load(W/'work/first.npy')
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
base_cv = fast.curves()
regions = {r for c in base_cv.values() for r in c.pred_region} | {r for c in base_cv.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10)
fold_of = {r: i for i, f in enumerate(folds) for r in f}
fid = np.array([fold_of[r] for r in reg])
fam = np.where(cls <= 1, 0, cls)
rng = np.random.default_rng(0); sub = rng.random(N)
mono = [1 if n in ('conf', 'logit') else 0 for n in names]
cat = [names.index(n) for n in ('cls', 'grp', 'mk') if n in names]
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, num_threads=args.threads, seed=0)
def train_predict(train_folds, pred_mask):
    """one model per family on train folds; returns scores for rows in pred_mask"""
    out = np.full(N, -1.0)
    for f in np.unique(fam[cls >= 0]):
        tr = (fam == f) & np.isin(fid, list(train_folds)) & pool
        neg_rate = 1.0
        tr &= y | (sub < neg_rate)
        w = np.where(y[tr], 1.0, 1.0 / neg_rate)
        pm = pred_mask & (fam == f) & pool
        if y[tr].sum() < 5:
            out[pm] = conf[pm]; continue
        ds = lgb.Dataset(X[tr], y[tr].astype(float), weight=w, categorical_feature=cat, free_raw_data=True)
        m = lgb.train(P, ds, args.rounds)
        out[pm] = m.predict(X[pm])
    return out
cache = W/f'work/nested_{args.tag}.npz'
K = len(folds)
if cache.exists():
    z = np.load(cache); outer = z['outer']; inner = z['inner']
else:
    outer = np.full(N, -1.0); inner = np.full((K, N), -1.0)
    for k in range(K):
        t = time.time()
        outer[fid == k] = train_predict(set(range(K)) - {k}, fid == k)[fid == k]
        for j in range(K):
            if j == k: continue
            inner[k][fid == j] = train_predict(set(range(K)) - {k, j}, fid == j)[fid == j]
        print(f'fold {k} {time.time()-t:.0f}s', flush=True)
    np.savez_compressed(cache, outer=outer, inner=inner)
nan = cls < 0
def honest_with(score_for_fold, mode_for_fold=None):
    counts = {c: {} for c in CL}; info = {'unseen_dropped': 0}
    for k, held in enumerate(folds):
        S = score_for_fold(k)
        cv = fast.curves(score=S)
        train = regions - held
        res = sweep_groups({c: curve_where(v, train) for c, v in cv.items()})
        taus = {}
        for c, v in res.items():
            for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
        for c, v in cv.items():
            hc = curve_where(v, held); per = counts[c]
            for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0: continue
            tau = np.array([taus.get(g, {}).get(c, np.inf) for g in hc.pred_group])
            info['unseen_dropped'] += int(np.isinf(tau).sum())
            kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]], counts, info
def mix(k, use_model_cls):
    S = conf.copy()
    trm = fid != k
    for c in use_model_cls:
        m = (cls == c)
        S[m & trm] = inner[k][m & trm]; S[m & ~trm] = outer[m & ~trm]
    return S
def report(name, s, counts, bc):
    per, bper = score_counts(counts)[1], score_counts(bc)[1]
    bt = paired_bootstrap_counts(counts, bc, site_family, 1000)
    print(f"{name}: all {s[0]:.6f} new {s[1]:.6f} opt {s[2]:.6f} | boot mean {bt['mean']:+.4f} [{bt['p5']:+.4f};{bt['p95']:+.4f}] P {bt['p_gt0']:.3f}", flush=True)
    print('   ' + ' '.join(f"{c[:6]} {per[c]['f1']-bper[c]['f1']:+.4f}" for c in CL), flush=True)
    return bt

def honest_select(level):
    """per outer fold choose base/model per class (level='class') or per (class,group) greedily, on train regions
    using inner OOF scores; apply choice to held fold with outer scores."""
    counts = {c: {} for c in CL}; choices = []
    base_cv = fast.curves()
    for k, held in enumerate(folds):
        train = regions - held
        def trainF1(S):
            res = sweep_groups({c: curve_where(v, train) for c, v in fast.curves(score=S).items()})
            return {c: res[c]['f1'] for c in res}
        fb = trainF1(conf); S_in = mix(k, range(len(CL))); fm = trainF1(S_in)
        use = {c: fm.get(c, 0) > fb.get(c, 0) + 0.002 for c in CL}
        if level == 'group':
            sel = np.zeros(N, bool)
            for ci, c in enumerate(CL):
                if not use[c]: continue
                best = fb[c]
                for g in sorted(set(grp[cls == ci])):
                    trial = sel | ((cls == ci) & (grp == g))
                    S = conf.copy(); S[trial] = S_in[trial]
                    f = trainF1(S)[c]
                    if f > best + 0.001: best, sel = f, trial
            S_tr = conf.copy(); S_tr[sel] = S_in[sel]
            S_out = conf.copy(); hm = sel & (fid == k); S_out[hm] = outer[hm]
            S_fin = np.where(fid == k, S_out, S_tr)
            choices.append(sorted({f"{CL[c][:6]}:{g}" for c, g in zip(cls[sel], grp[sel])}))
        else:
            S_fin = mix(k, [fast.cid[c] for c in CL if use[c]]); choices.append([c for c in CL if use[c]])
        cv = fast.curves(score=S_fin)
        res = sweep_groups({c: curve_where(v, train) for c, v in cv.items()})
        taus = {}
        for c, v in res.items():
            for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
        for c, v in cv.items():
            hc = curve_where(v, held); per = counts[c]
            for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0: continue
            tau = np.array([taus.get(g, {}).get(c, np.inf) for g in hc.pred_group])
            kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        print('fold', k, choices[-1], flush=True)
    return [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, optical)[0]], counts, choices
grp = T['grp']
b_s, b_counts, b_info = honest_with(lambda k: conf)
print('base', b_s, b_info)
res = {'base': b_s}
s, cnt, info = honest_with(lambda k: mix(k, range(len(CL)))); res['all_model'] = (s, report('model all classes', s, cnt, b_counts))
for c in range(len(CL)):
    s, cnt, info = honest_with(lambda k: mix(k, [c])); res[CL[c]] = (s, report(f'model only {CL[c]}', s, cnt, b_counts))
s, cnt, ch = honest_select('class'); res['select_class'] = (s, report('select per class', s, cnt, b_counts), ch)
pickle.dump(cnt, open(W/f'work/counts_{args.tag}_class.pkl', 'wb'))
if args.pergroup:
    s, cnt, ch = honest_select('group'); res['select_group'] = (s, report('select per group', s, cnt, b_counts), ch)
    pickle.dump(cnt, open(W/f'work/counts_{args.tag}_group.pkl', 'wb'))
json.dump(res, open(W/f'work/nested_{args.tag}.json', 'w'), indent=1, default=float)
