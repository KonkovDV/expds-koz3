"""Cheap per-candidate features from the stack table (no rasters)."""
from common import *
import fast, numpy as np
from scipy.spatial import cKDTree
T = fast.T; cls=T['cls'].astype(int); conf=T['conf']; grp=T['grp']; reg=T['reg']; CL=fast.CL; N=len(cls)
twin = np.load(W/'work/twin.npy')
boxes = json.loads((ROOT/'notes'/'las_boxes_59.json').read_text(encoding='utf-8'))
F = {}
lg = lambda x: np.log(np.clip(x,1e-6,1-1e-6)/(1-np.clip(x,1e-6,1-1e-6)))
F['conf'] = conf; F['logit'] = lg(conf); F['cls'] = cls.astype(float)
G = sorted(set(grp)); F['grp'] = np.searchsorted(G, grp).astype(float)
MK = sorted(set(T['mk'])); F['mk'] = np.searchsorted(MK, T['mk']).astype(float)
A, Pm = T['area'], T['perim']
F['area'] = np.log1p(A); F['compact'] = 4*np.pi*A/np.maximum(Pm,1e-6)**2
F['bw'] = np.log1p(np.maximum(T['bw'],T['bh'])); F['aspect'] = np.minimum(T['bw'],T['bh'])/np.maximum(np.maximum(T['bw'],T['bh']),1e-6)
F['nv'] = np.log1p(T['nv'])
ht = twin >= 0
F['has_twin'] = ht.astype(float); F['twin_conf'] = np.where(ht, conf[np.maximum(twin,0)], -1)
F['twin_diff'] = np.where(ht, conf - conf[np.maximum(twin,0)], 0)
F['has_las'] = np.isin(reg, list(boxes)).astype(float)
# region x group x class stats
key = np.char.add(np.char.add(reg.astype(str), '|'), np.char.add(grp.astype(str), '|' ) )
key = np.char.add(key, cls.astype(str))
uk, inv = np.unique(key, return_inverse=True)
cnt = np.bincount(inv); F['rg_n'] = np.log1p(cnt[inv])
pct = np.zeros(N); rmax = np.zeros(N); rmed = np.zeros(N); n_hi = np.zeros(N)
order = np.lexsort((-conf, inv))
starts = np.r_[0, np.cumsum(cnt)[:-1]]
rank = np.empty(N); rank[order] = np.arange(N) - starts[inv[order]]
F['rg_rank'] = np.log1p(rank); F['rg_pct'] = rank/np.maximum(cnt[inv]-1,1)
mx = np.full(len(uk), -1.0); np.maximum.at(mx, inv, conf); F['rg_max'] = mx[inv]; F['conf_rel'] = conf/np.maximum(mx[inv],1e-6)
sm = np.bincount(inv, conf); F['rg_mean'] = sm[inv]/cnt[inv]
# global group-class quantile of conf
gk = np.char.add(grp.astype(str), cls.astype(str)); ug, gi = np.unique(gk, return_inverse=True)
q = np.zeros(N)
for g in range(len(ug)):
    ii = np.flatnonzero(gi==g); o = np.argsort(conf[ii], kind='stable'); qq = np.empty(len(ii)); qq[o] = np.arange(len(ii))/max(len(ii)-1,1); q[ii]=qq
F['gq'] = q
# spatial neighbourhood: per region, kd-tree over candidates with gq>=0.9 ("strong") of each family
fam = np.where(cls<=1, 0, cls)  # kurgans merged
strong = q >= 0.98
for nm in ['nb_same10','nb_same30','nb_same100','nb_strong30','nb_strong100','og_max5','og_n5','nn_dist','nb_otherfam50']: F[nm]=np.zeros(N)
X = np.c_[T['cx'], T['cy']]
for r in np.unique(reg):
    ir = np.flatnonzero(reg==r)
    for f_ in np.unique(fam[ir]):
        ii = ir[fam[ir]==f_]
        tr = cKDTree(X[ii])
        # unique locations to avoid twins counting: use same class only
        for rad, nm in ((10,'nb_same10'),(30,'nb_same30'),(100,'nb_same100')):
            F[nm][ii] = np.log1p(tr.query_ball_point(X[ii], rad, return_length=True))
        d, _ = tr.query(X[ii], k=min(3, len(ii)))
        d = np.atleast_2d(d)
        # nearest other location (>0.5 m)
        dd = np.where(d > 0.5, d, np.inf).min(1) if d.shape[1]>1 else np.full(len(ii), np.inf)
        F['nn_dist'][ii] = np.log1p(np.minimum(dd, 1e4))
        js = ii[strong[ii]]
        if len(js):
            ts = cKDTree(X[js])
            for rad, nm in ((30,'nb_strong30'),(100,'nb_strong100')):
                F[nm][ii] = np.log1p(ts.query_ball_point(X[ii], rad, return_length=True))
        # other-group agreement within 5 m: max gq among candidates of other groups
        lst = tr.query_ball_point(X[ii], 5.0)
        gg = gi[ii]; gr = grp[ii]
        omax = np.zeros(len(ii)); on = np.zeros(len(ii))
        for a, nb in enumerate(lst):
            if len(nb) <= 1: continue
            nb = np.asarray(nb); m = gr[nb] != gr[a]
            if m.any():
                omax[a] = q[ii[nb[m]]].max(); on[a] = len(set(gr[nb[m]]))
        F['og_max5'][ii] = omax; F['og_n5'][ii] = on
    # other family strong within 50 m
    for f_ in np.unique(fam[ir]):
        ii = ir[fam[ir]==f_]; js = ir[(fam[ir]!=f_) & strong[ir]]
        if len(js): F['nb_otherfam50'][ii] = np.log1p(cKDTree(X[js]).query_ball_point(X[ii], 50, return_length=True))
names = list(F); Xf = np.c_[tuple(F[n] for n in names)].astype(np.float32)
np.save(W/'work/X.npy', Xf); json.dump(names, open(W/'work/X_names.json','w'))
okc = np.zeros((N, len(CL)), bool)
for (pc, gc), (P, J, S) in T['pairs'].items(): okc[P, fast.cid[gc]] = True
np.save(W/'work/okc.npy', okc)
print(Xf.shape, names)
