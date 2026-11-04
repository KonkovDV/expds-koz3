#!/usr/bin/env python3
"""Q6 step 2: LightGBM OOF by honest-10 folds -> ex_fgbm.geojson (PREREG 870433c)."""
import sys, json, pickle
sys.path.insert(0, '/workspace/koz3/w4/repo')
from pathlib import Path
import numpy as np, lightgbm as lgb
from shapely.geometry import mapping
from expds import finds_cand as F
from expds.raster import circle_3857
W = Path('/workspace/koz3/orch/bot-W4/work/q6')
T = json.load(open('/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json'))
fold_of = {r: i for i, f in enumerate(T['folds']) for r in f}
R = {p.stem: pickle.load(open(p, 'rb')) for p in sorted((W / 'cache').glob('*.pkl'))}
gt = json.loads(open('/workspace/koz3/w5data/gt_train_v2.geojson', 'rb').read())
hasgt = {f['properties']['region_name'] for f in gt['features'] if f['properties']['class_name'] == 'finds_points'}
PRM = dict(objective='binary', num_leaves=31, learning_rate=0.05, min_child_samples=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=0, num_threads=4, verbose=-1)
def data(regs):
    X, y = [], []
    for r in regs:
        for d in R[r]:
            k = d['kept']; X.append(d['X'][k]); y.append((d['dist'][k] <= 1.0).astype(int))
    return np.concatenate(X), np.concatenate(y)
feats, diag = [], {}
for fo in sorted({fold_of[r] for r in R}):
    test = [r for r in R if fold_of[r] == fo]
    train = [r for r in R if fold_of[r] != fo and r in hasgt]
    X, y = data(train)
    m = lgb.train(PRM, lgb.Dataset(X, y, feature_name=list(F.NAMES)), 400)
    m.save_model(str(W / f'model_fold{fo}.txt'))
    for r in test:
        for d in R[r]:
            k = np.flatnonzero(d['kept'])
            if not len(k): continue
            p = m.predict(d['X'][k])
            keep = k[F.nms(d['xy'][k], p, 1.0)]
            pk = m.predict(d['X'][keep])
            yk = d['dist'][keep] <= 1.0
            diag[r] = dict(n_cand=int(len(k)), n_out=int(len(keep)), pos=int(yk.sum()))
            for (x, yy), c in zip(d['xy'][keep], pk):
                g = circle_3857(float(x), float(yy), 0.5, d['crs'])
                feats.append({'type': 'Feature', 'geometry': mapping(g), 'properties': {
                    'class_name': 'finds_points', 'confidence': float(c), 'region_name': r, 'markup_type': 'maggr',
                    'threshold_group': 'maggr_fgbm', 'source': d['source']}})
    print('fold', fo, 'train', len(train), 'n', len(y), 'pos', int(y.sum()), 'test', test, flush=True)
json.dump({'type': 'FeatureCollection', 'features': feats}, open(W / 'ex_fgbm.geojson', 'w'), ensure_ascii=False)
json.dump(diag, open(W / 'oof_diag.json', 'w'), ensure_ascii=False, indent=1)
print('wrote', len(feats), diag)
