"""W0 seam audit M1: anchor/shape loss on the OOF stack (official _match_quality semantics).
For every polygon pred vs same-class GT polygons: current hit (from W5 table pairs),
rp-in-GT, centroid(in P)-in-GT, polylabel-in-GT. Writes counts + new pair lists for disk-replacement variants."""
import pickle, sys, json, numpy as np, shapely, time
from shapely.strtree import STRtree
from shapely.ops import polylabel
W='/workspace/koz3/w5data/work'
T=pickle.load(open(f'{W}/table.pkl','rb'))
CL=T['CL']; cid={c:i for i,c in enumerate(CL)}
KUR=['kurgany_povrezhdennye','kurgany_tselye']
gwkb=pickle.load(open(f'{W}/gt_wkb.pkl','rb'))
G={c:shapely.from_wkb(gwkb[c]) if len(gwkb[c]) else np.array([],object) for c in CL}
wkb=pickle.load(open(f'{W}/geom_wkb.pkl','rb'))
N=len(wkb); print('N',N,flush=True)
cls=T['cls']
# current hit set per (pc,gc): set of preds with any ok pair
cur={k:set(map(int,np.unique(v[0]))) for k,v in T['pairs'].items()}
R=0.05  # disk radius, metres (UTM)
out={}; newpairs={a:{} for a in ('rp','cen','pl')}
for pc in CL:
    idx=np.flatnonzero(cls==cid[pc]); 
    if not len(idx): continue
    for gc in (KUR if pc in KUR else [pc]):
        gg=G[gc]
        if not len(gg): continue
        ispt=shapely.get_type_id(gg)==0
        tree=STRtree([x.buffer(2.0) if t else x for x,t in zip(gg,ispt)])
        st=dict(n=int(len(idx)),cur=0,rp=0,cen=0,pl=0,rp_not_cur=0,cur_not_rp=0,cen_not_rp=0,rp_not_cen=0,pl_not_rp=0,rp_not_pl=0,n_gt_pt=int(ispt.sum()),n_gt=len(gg))
        P={a:[] for a in newpairs}; J={a:[] for a in newpairs}
        for s in range(0,len(idx),20000):
            ii=idx[s:s+20000]; geoms=shapely.from_wkb(wkb[ii])
            ok=np.array([g is not None and g.geom_type=='Polygon' for g in geoms])
            rp=np.array([g.representative_point() if o else None for g,o in zip(geoms,ok)],object)
            ce=shapely.centroid(geoms)
            inside=np.array([o and g.covers(c) for g,c,o in zip(geoms,ce,ok)])
            ce=np.where(inside,ce,rp)
            pl=[]
            for g,o in zip(geoms,ok):
                if not o: pl.append(None); continue
                try: pl.append(polylabel(g,tolerance=max(0.01,np.sqrt(g.area)/200)))
                except Exception: pl.append(g.representative_point())
            pl=np.array(pl,object)
            anc={'rp':rp,'cen':ce,'pl':pl}
            hits={}
            for a,pts in anc.items():
                m=ok.copy()
                q,j=tree.query(np.where(m,pts,None))
                h=np.zeros(len(ii),bool)
                for qq,jj in zip(q,j):
                    p=pts[qq]; g=gg[jj]
                    if ispt[jj]:
                        good=p.distance(g)<=2.0
                    else:
                        good=g.contains(p) and g.exterior.distance(p)>=R*0.71  # disk mostly inside -> cov>=0.5 & anchor inside
                    if good:
                        h[qq]=True; P[a].append(int(ii[qq])); J[a].append(int(jj))
                hits[a]=h
            c0=np.array([int(p) in cur.get((pc,gc),()) for p in ii])
            st['cur']+=int(c0.sum())
            for a in anc: st[a]+=int(hits[a].sum())
            st['rp_not_cur']+=int((hits['rp']&~c0).sum()); st['cur_not_rp']+=int((c0&~hits['rp']).sum())
            st['cen_not_rp']+=int((hits['cen']&~hits['rp']).sum()); st['rp_not_cen']+=int((hits['rp']&~hits['cen']).sum())
            st['pl_not_rp']+=int((hits['pl']&~hits['rp']).sum()); st['rp_not_pl']+=int((hits['rp']&~hits['pl']).sum())
        for a in newpairs:
            Pa=np.array(P[a],np.int64); Ja=np.array(J[a],np.int64)
            newpairs[a][(pc,gc)]=(Pa,Ja,np.ones(len(Pa)))  # disk inside GT -> cov=1
        out[f'{pc}|{gc}']=st; print(pc,gc,st,flush=True)
json.dump(out,open('m1_anchor.json','w'),indent=1)
pickle.dump(newpairs,open(f'/workspace/koz3/w5data/work/w0_disk_pairs.pkl','wb'))
print('done')
