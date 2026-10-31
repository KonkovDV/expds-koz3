"""Монте-Карло: P(попадание) по правилу метрики (cov>=0.5 к меньшей фигуре И representative_point в эталоне)
для разных форм/размеров предсказания при шуме центра и размера.
Эталоны: (a) синтетика круг/эллипс, (b) реальные эталоны классов (UTM), нормированные на r_eq."""
from common7 import *
import expds.metric as _m
import numpy as np, shapely, json, sys
from shapely import affinity
rng=np.random.default_rng(0)
N=int(sys.argv[1]) if len(sys.argv)>1 else 1500

def match(P,G):
    inter=shapely.area(shapely.intersection(P,G))
    cov=inter/np.minimum(shapely.area(P),shapely.area(G))
    rp=shapely.point_on_surface(P)
    return (cov>=0.5)&shapely.covers(G,rp)

unit=shapely.Point(0,0).buffer(1,quad_segs=16)
def pred_shape(kind,s,theta):
    """фигура площади pi*s^2 (как круг радиуса s), центр (0,0)"""
    if kind=='circle': g=unit
    elif kind=='ellipse2': g=affinity.scale(unit,np.sqrt(2),1/np.sqrt(2))
    elif kind=='square': a=np.sqrt(np.pi)/2; g=shapely.box(-a,-a,a,a)
    elif kind=='rect1x3': a=np.sqrt(np.pi/3)/2; g=shapely.box(-3*a,-a,3*a,a)
    g=affinity.rotate(g,theta,origin=(0,0),use_radians=True)
    return affinity.scale(g,s,s,origin=(0,0))

def run(Gs, kinds, sizes, sigmas, size_cv=0.3, buffers=()):
    """Gs: list of GT polygons normalized (centroid at 0, r_eq=1)."""
    res={}
    idx=rng.integers(0,len(Gs),N)
    G=np.array([Gs[i] for i in idx],dtype=object)
    th=rng.uniform(0,np.pi,N)
    for sig in sigmas:
        off=rng.normal(0,sig,(N,2))   # изотропный гаусс, sigma в долях r_eq
        lns=np.exp(rng.normal(0,size_cv,N))   # мультипликативный шум размера
        base_in=shapely.covers(G,shapely.points(off)).mean()  # верхняя граница: центр в эталоне
        res[f'sig={sig}']={'P(center in GT)':round(float(base_in),3)}
        for k in kinds:
            for s in sizes:
                P=np.array([affinity.translate(pred_shape(k,s*l,t),x,y) for l,t,(x,y) in zip(lns,th,off)],dtype=object)
                res[f'sig={sig}'][f'{k} s={s}']=round(float(match(P,G).mean()),3)
        for b in buffers:   # сдвинутый эталон, раздутый/сжатый буфером b*r_eq (b<0 — эрозия)
            P=np.array([affinity.translate(g,x,y).buffer(b) for g,(x,y) in zip(G,off)],dtype=object)
            ok=~shapely.is_empty(P)
            m=np.zeros(N,bool); m[ok]=match(P[ok],G[ok])
            res[f'sig={sig}'][f'GTshape buf={b}']=round(float(m.mean()),3)
    return res

def norm(g):
    c=g.centroid; R=np.sqrt(g.area/np.pi)
    return affinity.scale(affinity.translate(g,-c.x,-c.y),1/R,1/R,origin=(0,0))

out={}
sizes=[0.02,0.1,0.25,0.5,0.75,1.0,1.5,2.0,4.0]
kinds=['circle','ellipse2','square','rect1x3']
sig=[0.1,0.25,0.5,0.75]
out['synthetic_circle']=run([unit],kinds,sizes,sig,buffers=[-0.5,-0.25,0,0.5,1.0])
out['synthetic_ellipse1.5']=run([affinity.scale(unit,np.sqrt(1.5),1/np.sqrt(1.5))],kinds,sizes,sig,buffers=[-0.5,0,0.5])
mod=_m.load_official(METRIC); calc=mod.F1Calculator()
gt=load('/workspace/rerank/gt_train_v2.geojson'); G=_m._collect(mod,calc,gt,True)
for c in ['kurgany_povrezhdennye','kurgany_tselye','gorodishcha','object_poly']:
    gs=[norm(x[0]) for x in G[c] if x[0].geom_type=='Polygon' and x[0].area>0]
    out['GT_'+c]=run(gs,['circle'],sizes,sig,buffers=[-0.25,0,0.5])
json.dump(out,open(W/'mc_shapes.json','w'),indent=1,ensure_ascii=False)
for k,v in out.items():
    print('\n###',k)
    for s,d in v.items(): print(' ',s,d)
