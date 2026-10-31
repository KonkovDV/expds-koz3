from common7 import *
import expds.metric as _m
import numpy as np, shapely
from collections import defaultdict
mod=_m.load_official(METRIC); calc=mod.F1Calculator()
st=load(W/'stack_sub6000.geojson'); gt=load('/workspace/rerank/gt_train_v2.geojson')
P=_m._collect(mod,calc,st,False); G=_m._collect(mod,calc,gt,True)
def q(a): a=np.asarray(a,float); return ' '.join(f'{x:.2f}' for x in np.percentile(a,[5,25,50,75,95])) if len(a) else '-'
CL=['kurgany_povrezhdennye','kurgany_tselye','gorodishcha','object_poly','arkhitektury','fortifikatsii','finds_points']
out={}
for c in CL:
    g=[x[0] for x in G.get(c,[])]
    types=defaultdict(int)
    for x in g: types[x.geom_type]+=1
    pol=[x for x in g if x.geom_type in('Polygon','MultiPolygon')]
    A=np.array([x.area for x in pol]); req=np.sqrt(A/np.pi)
    comp=np.array([4*np.pi*x.area/x.length**2 for x in pol]) if pol else []
    # elongation via min rotated rectangle
    el=[]
    for x in pol:
        r=x.minimum_rotated_rectangle; cs=np.array(r.exterior.coords)
        e=[np.hypot(*(cs[1]-cs[0])),np.hypot(*(cs[2]-cs[1]))]; el.append(max(e)/max(min(e),1e-6))
    # centroid in polygon?
    cin=np.mean([x.contains(x.centroid) for x in pol]) if pol else float('nan')
    # nearest-neighbour gap same region
    byr=defaultdict(list)
    for x in G.get(c,[]): byr[x[2]].append(x[0])
    gaps=[];cd=[]
    for r,gs in byr.items():
        if len(gs)<2: continue
        tr=shapely.STRtree(gs)
        for i,x in enumerate(gs):
            idx=tr.query_nearest(x,exclusive=True,all_matches=False)
            if len(idx)==0: continue
            j=int(idx[0]); gaps.append(x.distance(gs[j])); cd.append(x.centroid.distance(gs[j].centroid))
    print(f'\n== GT {c}: n={len(g)} types={dict(types)}')
    if pol:
        print(f' area p5..p95: {q(A)}\n r_eq: {q(req)}\n compact4piA/P2: {q(comp)}\n elong(MRR): {q(el)}\n centroid inside: {cin:.3f}')
    print(f' NN gap m: {q(gaps)}  share gap==0: {np.mean(np.array(gaps)==0) if gaps else 0:.3f}\n NN centroid dist: {q(cd)}')
    # preds by group
    pg=defaultdict(list)
    for x in P.get(c,[]): pg[x[3]].append(x)
    for grp,xs in sorted(pg.items()):
        xs=sorted(xs,key=lambda t:-t[1])[:2000]
        A=np.array([t[0].area for t in xs]); multi=np.mean([t[0].geom_type!='Polygon' for t in xs]); nv=[shapely.get_num_coordinates(t[0]) for t in xs]
        print(f'  pred {grp:10s} top{len(xs)} r_eq {q(np.sqrt(A/np.pi))} multi={multi:.2f} nverts med {np.median(nv):.0f}')
