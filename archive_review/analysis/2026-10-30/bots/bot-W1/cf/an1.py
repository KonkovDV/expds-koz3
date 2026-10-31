from common import *
import fast, numpy as np
from scipy.spatial import cKDTree
T = fast.T; cls=T['cls']; conf=T['conf']; grp=T['grp']; reg=T['reg']
P, Tq = fast.cid['kurgany_povrezhdennye'], fast.cid['kurgany_tselye']
kur = np.flatnonzero((cls==P)|(cls==Tq))
twin = np.full(len(cls), -1)
key = np.char.add(np.char.add(reg[kur].astype(str), '|'), grp[kur].astype(str))
for k in np.unique(key):
    ii = kur[key==k]; a = ii[cls[ii]==P]; b = ii[cls[ii]==Tq]
    if not len(a) or not len(b): continue
    tr = cKDTree(np.c_[T['cx'][b], T['cy'][b]]); d, j = tr.query(np.c_[T['cx'][a], T['cy'][a]])
    ok = d < 0.5; twin[a[ok]] = b[j[ok]]; twin[b[j[ok]]] = a[ok]
np.save(W/'work/twin.npy', twin)
# hits per class under base
pos = {}
for (pc, gc), (Pp, J, S) in T['pairs'].items(): pos.setdefault(gc, set()).update(Pp.tolist())
for g in np.unique(grp[kur]):
    for c in (P, Tq):
        m = (grp==g)&(cls==c); n=m.sum(); tw=(twin[m]>=0).mean()
        ii=np.flatnonzero(m&(twin>=0)); dc = conf[ii]-conf[twin[ii]] if len(ii) else np.array([0.])
        ok_p = np.isin(np.flatnonzero(m), list(pos['kurgany_povrezhdennye'])).sum(); ok_t=np.isin(np.flatnonzero(m), list(pos['kurgany_tselye'])).sum()
        print(f"{g:10s} {fast.CL[c][8:12]} n={n:7d} twin={tw:.3f} conf-twin mean={dc.mean():+.3f} absmax={np.abs(dc).max():.3f} okP={ok_p} okT={ok_t}")
