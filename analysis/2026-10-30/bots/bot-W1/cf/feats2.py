"""Extra features: NMS/duplicate context; pool mask; first-hit labels."""
from common import *
import fast, numpy as np
from scipy.spatial import cKDTree
T = fast.T; cls=T['cls'].astype(int); conf=T['conf']; grp=T['grp']; reg=T['reg']; CL=fast.CL; N=len(cls)
X = np.load(W/'work/X.npy'); names = json.load(open(W/'work/X_names.json'))
gq = X[:, names.index('gq')]
XY = np.c_[T['cx'], T['cy']]
E = {k: np.zeros(N, np.float32) for k in ['d_nhigher_g','d_maxq_other_g','d_localmax','d_maxq_othersub','d_n8']}
for r in np.unique(reg):
    ir = np.flatnonzero(reg==r)
    for c in np.unique(cls[ir]):
        if c < 0: continue
        ii = ir[cls[ir]==c]; tr = cKDTree(XY[ii]); lst = tr.query_ball_point(XY[ii], 8.0)
        gi = grp[ii]; qi = gq[ii]
        for a, nb in enumerate(lst):
            nb = np.asarray(nb); nb = nb[nb != a]
            E['d_n8'][ii[a]] = len(nb)
            if not len(nb): E['d_localmax'][ii[a]] = 1; continue
            same = gi[nb] == gi[a]
            E['d_nhigher_g'][ii[a]] = (qi[nb[same]] > qi[a]).sum()
            E['d_maxq_other_g'][ii[a]] = qi[nb[~same]].max() if (~same).any() else 0
            E['d_localmax'][ii[a]] = float(qi[a] >= qi[nb].max())
        if c in (0, 1):
            jj = ir[cls[ir]==1-c]
            if len(jj):
                t2 = cKDTree(XY[jj]); l2 = t2.query_ball_point(XY[ii], 8.0)
                E['d_maxq_othersub'][ii] = [gq[jj[np.asarray(l)]].max() if l else 0 for l in l2]
X2 = np.c_[X, np.c_[tuple(E[k] for k in E)]].astype(np.float32); names2 = names + list(E)
np.save(W/'work/X2.npy', X2); json.dump(names2, open(W/'work/X2_names.json','w'))
# pool: top 15% (>=1000) per (group,class); all for small groups
pool = np.zeros(N, bool)
key = np.char.add(grp.astype(str), cls.astype(str))
for k in np.unique(key):
    ii = np.flatnonzero(key==k)
    if len(ii) <= 5000: pool[ii] = True; continue
    m = max(1000, int(0.15*len(ii))); o = ii[np.argsort(-conf[ii], kind='stable')[:m]]; pool[o] = True
pool &= cls >= 0
# first-hit label under base ordering
cv = fast.curves(); first = np.zeros(N, bool)
for c in CL:
    idx = np.flatnonzero(cls == fast.cid[c]); o = idx[np.argsort(-conf[idx], kind='stable')]; first[o[cv[c].hit]] = True
np.save(W/'work/pool.npy', pool); np.save(W/'work/first.npy', first)
okc = np.load(W/'work/okc.npy'); own = okc[np.arange(N), np.maximum(cls,0)] & (cls>=0)
print('pool', pool.sum(), 'own in pool', (own&pool).sum(), '/', own.sum(), 'first in pool', (first&pool).sum(), '/', first.sum())
for k in np.unique(key):
    ii = key==k; print(k, ii.sum(), (ii&pool).sum(), (own&ii).sum(), (own&ii&pool).sum(), (first&ii).sum())
