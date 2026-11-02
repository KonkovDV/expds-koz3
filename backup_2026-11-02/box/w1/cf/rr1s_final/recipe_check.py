import os, sys; sys.path.insert(0, os.getcwd())
from common import *
import numpy as np, lightgbm as lgb, pickle, json
T = pickle.load(open(W/'work/table.pkl','rb')); cls=T['cls'].astype(int); reg=T['reg']; N=len(cls)
regions = set(map(str, reg)) | {r for c in T['CL'] for r in T['greg'][c]}
folds = family_folds(regions, site_family, 10); fo = {r:i for i,f in enumerate(folds) for r in f}; fid = np.array([fo[r] for r in reg])
X = np.load(W/'work/X4.npy'); names = json.load(open(W/'work/X4_names.json')); pool=np.load(W/'work/pool.npy'); y=np.load(W/'work/first.npy')
fam = np.where(cls<=1,0,cls); mono=[1 if n in('conf','logit') else 0 for n in names]; cat=[names.index(n) for n in ('cls','grp','mk')]
P = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, num_threads=4, seed=0)
z = np.load(W/'work/nested_w5cf.npz'); outer=z['outer']
for f in (0, 6):
    tr=(fam==f)&(fid!=0)&pool; pm=(fam==f)&(fid==0)&pool
    m=lgb.train(P, lgb.Dataset(X[tr], y[tr].astype(float), categorical_feature=cat), 300)
    print('fam', f, 'n', pm.sum(), 'max|diff| vs nested outer fold0', float(np.abs(m.predict(X[pm])-outer[pm]).max()))
