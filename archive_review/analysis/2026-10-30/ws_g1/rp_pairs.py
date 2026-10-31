"""Reproduce W0-seams 5171c35 m1_anchor 'rp' pairs (5 cm disk at representative_point) for gor/arch/obj/fort -> work/w0_disk_pairs.pkl {'rp': {...}}."""
import pickle, numpy as np, shapely
from shapely.strtree import STRtree
W='/workspace/koz3/w5data/work'
T=pickle.load(open(f'{W}/table.pkl','rb')); CL=T['CL']; cid={c:i for i,c in enumerate(CL)}
gwkb=pickle.load(open(f'{W}/gt_wkb.pkl','rb')); G={c:shapely.from_wkb(gwkb[c]) if len(gwkb[c]) else np.array([],object) for c in CL}
wkb=pickle.load(open(f'{W}/geom_wkb.pkl','rb')); cls=T['cls']; R=0.05; out={}
for pc in ('gorodishcha','arkhitektury','object_poly','fortifikatsii'):
    idx=np.flatnonzero(cls==cid[pc]); gg=G[pc]
    if not len(idx) or not len(gg): continue
    ispt=shapely.get_type_id(gg)==0; tree=STRtree([x.buffer(2.0) if t else x for x,t in zip(gg,ispt)])
    geoms=shapely.from_wkb(wkb[idx]); ok=np.array([g is not None and g.geom_type=='Polygon' for g in geoms])
    rp=np.array([g.representative_point() if o else None for g,o in zip(geoms,ok)],object)
    q,j=tree.query(rp); P=[];J=[]
    for qq,jj in zip(q,j):
        p=rp[qq]; g=gg[jj]
        good = p.distance(g)<=2.0 if ispt[jj] else (g.contains(p) and g.exterior.distance(p)>=R*0.71)
        if good: P.append(int(idx[qq])); J.append(int(jj))
    out[(pc,pc)]=(np.array(P,np.int64),np.array(J,np.int64),np.ones(len(P)))
    cur=len(np.unique(T['pairs'].get((pc,pc),(np.zeros(0),))[0])); print(pc,'rp hits',len(np.unique(P)),'cur',cur,flush=True)
pickle.dump({'rp':out},open(f'{W}/w0_disk_pairs.pkl','wb')); print('done')
