"""Вырезка object_poly и finds_points из /workspace/rerank/work/table.pkl + geom_wkb.pkl (стек stack_arch, UTM)."""
import pickle, numpy as np, gc
T = pickle.load(open('/workspace/rerank/work/table.pkl','rb'))
CL = T['CL']; keep = {}
for c in ['object_poly','finds_points']:
    k = CL.index(c); idx = np.flatnonzero(T['cls']==k)
    keep[c] = idx
out = dict(CL=CL, n_gt=T['n_gt'], greg={c:T['greg'][c] for c in keep})
for c, idx in keep.items():
    d = {f: T[f][idx] for f in ['conf','reg','grp','mk','area','cx','cy']}
    d['idx'] = idx
    P,J,S = T['pairs'][(c,c)]; d['pairs'] = (P,J,S)
    out[c] = d
del T; gc.collect()
G = pickle.load(open('/workspace/rerank/work/geom_wkb.pkl','rb'))
for c, idx in keep.items(): out[c]['wkb'] = G[idx]
del G; gc.collect()
gw = pickle.load(open('/workspace/rerank/work/gt_wkb.pkl','rb'))
out['gt_wkb'] = {c: gw[c] for c in keep}
pickle.dump(out, open('sub_geo.pkl','wb'), protocol=5)
print({c: len(v) for c,v in keep.items()})
