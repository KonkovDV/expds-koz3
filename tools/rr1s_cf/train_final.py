"""RR1s-cf: final rankers + inference reference for expds.w5cf_infer (offline, from the W5 training table).

Same learner as nested2.py --xset X4 --tag w5cf --label first (LightGBM per family: 0 = kurgans cls 0/1, else cls;
rows = pool; label = first; 300 rounds, seed 0), but trained on ALL train regions (all 10 folds).
--nocnx: cnx_* columns set to NaN before training (the run has no W2 ConvNeXt seeds → cnx_* are NaN at inference).

Inputs: W5_DATA/work/{X4.npy,X4_names.json,pool.npy,first.npy,table.pkl}.
Outputs (--out): fam<k>.txt, meta.json (feature order, encodings, pool thresholds, versions, sha256), gq_ref.npz
(per (grp, cls): unique train conf values + counts, for the global group-class quantile gq).
  W5_DATA=/workspace/koz3/w1/cf/w5d python tools/rr1s_cf/train_final.py --out models/w5cf_infer [--nocnx]
"""
import argparse, hashlib, json, os, pickle, platform, time
from pathlib import Path
import numpy as np, lightgbm as lgb

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--rounds', type=int, default=300)
ap.add_argument('--threads', type=int, default=4); ap.add_argument('--nocnx', action='store_true'); a = ap.parse_args()
W = Path(os.environ.get('W5_DATA', '/workspace/koz3/w1/cf/w5d')); O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
T = pickle.load(open(W / 'work/table.pkl', 'rb'))
cls = T['cls'].astype(int); grp = np.asarray(T['grp']).astype(str); mk = np.asarray(T['mk']).astype(str); conf = np.asarray(T['conf'], float)
CL = list(T['CL'])
X = np.load(W / 'work/X4.npy'); names = json.load(open(W / 'work/X4_names.json'))
if a.nocnx:
    X[:, [i for i, n in enumerate(names) if n.startswith('cnx_')]] = np.nan
pool = np.load(W / 'work/pool.npy'); y = np.load(W / 'work/first.npy'); N = len(cls); assert X.shape == (N, len(names))
fam = np.where(cls <= 1, 0, cls)
mono = [1 if n in ('conf', 'logit') else 0 for n in names]
cat = [names.index(n) for n in ('cls', 'grp', 'mk') if n in names]
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, num_threads=a.threads, seed=0)

# pool rule of feats2.py: per (grp, cls) top 15 % (>= 1000) by conf; groups <= 5000 rows whole → conf threshold
pool_thr, gq_keys, gq_off, gq_val, gq_cnt = {}, [], [0], [], []
for g in sorted(set(grp)):
    for c in sorted(set(cls[grp == g])):
        ii = np.flatnonzero((grp == g) & (cls == c)); n = len(ii)
        if n <= 5000:
            thr = -np.inf
        else:
            m = max(1000, int(0.15 * n)); thr = float(np.sort(conf[ii])[::-1][m - 1])
        pool_thr[f'{g}|{c}'] = thr if np.isfinite(thr) else None
        v, k = np.unique(conf[ii], return_counts=True)
        gq_keys.append(f'{g}|{c}'); gq_val.append(v); gq_cnt.append(k); gq_off.append(gq_off[-1] + len(v))
np.savez_compressed(O / 'gq_ref.npz', keys=np.array(gq_keys), off=np.array(gq_off, np.int64),
                    val=np.concatenate(gq_val), cnt=np.concatenate(gq_cnt).astype(np.int64))
meta = dict(variant='nocnx' if a.nocnx else 'full', names=names, categorical=[names[i] for i in cat], rounds=a.rounds,
            params={k: v for k, v in P.items() if k != 'monotone_constraints'}, monotone=[n for n, m in zip(names, mono) if m],
            CL=CL, grp_codes=sorted(set(grp)), mk_codes=sorted(set(mk)), family_of_cls={c: (0 if i <= 1 else i) for i, c in enumerate(CL)},
            pool_thr=pool_thr, pool_rule='conf >= pool_thr[grp|cls] (null = whole group); unseen (grp, cls) → not in pool',
            versions=dict(lightgbm=lgb.__version__, numpy=np.__version__, python=platform.python_version()),
            train=dict(rows=int(N), pool=int(pool.sum()), regions=int(len(set(T['reg'])))), models={}, fallback_conf=[])
for f in np.unique(fam[cls >= 0]):
    tr = (fam == f) & pool
    if y[tr].sum() < 5: meta['fallback_conf'].append(int(f)); continue
    t = time.time(); m = lgb.train(P, lgb.Dataset(X[tr], y[tr].astype(float), categorical_feature=cat, free_raw_data=True), a.rounds)
    p = O / f'fam{int(f)}.txt'; m.save_model(str(p))
    meta['models'][int(f)] = dict(file=p.name, rows=int(tr.sum()), pos=int(y[tr].sum()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    print('fam', f, CL[f] if f else 'kurgany', tr.sum(), y[tr].sum(), f'{time.time()-t:.0f}s', flush=True)
json.dump(meta, open(O / 'meta.json', 'w'), ensure_ascii=False, indent=1)
print('done', sorted(meta['models']), 'fallback', meta['fallback_conf'])
