"""bot-7. Шаг 0: сколько городищ (73 без лидара) имеют кандидата в существующем стеке при любом пороге."""
import json, collections, numpy as np, shapely, pickle
from common import *
T,G,greg,gmk,lid,mod=load_gt(); calc=mod.F1Calculator()
S=load_scenes()
k=T['CL'].index(C); rows=np.flatnonzero(T['cls']==k)
W=pickle.load(open('/workspace/rerank/work/geom_wkb.pkl','rb')); geo={int(i):shapely.from_wkb(W[i]) for i in rows}; del W
NONLI={'sp_or','ae','or','sp_arch','sp_focal','sp_focal2','ae_fort','gor_phys'}
grp=T['grp']; cx=T['cx']; cy=T['cy']; reg=T['reg']
res=[]; 
for j in np.flatnonzero(~lid):
    g=G[j]; c=g.centroid; r=greg[j]
    sc=[s for s in S if s['reg']==r and s['poly'].intersects(g)]
    d=dict(j=int(j),reg=str(r),mk=str(gmk[j]),area=round(g.area),diam=round(2*np.sqrt(g.area/np.pi)),
           scenes=[(s['file'],s['kind'],round(s['poly'].area/1e4,1),round(Point(s['poly'].centroid).distance(c))) for s in sc])
    # кандидаты класса gor
    best=None
    for i in rows:
        if reg[i]!=r: continue
        gi=geo[int(i)]; ok=calc._match_quality(gi,g)[0]
        dist=gi.centroid.distance(c); key=(ok,-dist)
        if best is None or key>best[0]: best=(key,dict(grp=str(grp[i]),conf=round(float(T['conf'][i]),4),ok=bool(ok),dist=round(dist)))
    d['gor_best']=best[1] if best else None
    m=(reg==r)
    for nm,msk in (('any_nonli',m&np.isin(grp,list(NONLI))),('any',m)):
        ii=np.flatnonzero(msk)
        if len(ii): dd=np.hypot(cx[ii]-c.x,cy[ii]-c.y); q=ii[dd.argmin()]; d[nm]=dict(dist=round(float(dd.min())),cls=T['CL'][T['cls'][q]],grp=str(grp[q]),n50=int((dd<50).sum()),n100=int((dd<100).sum()))
        else: d[nm]=None
    res.append(d)
json.dump(res,open('g0_stack.json','w'),ensure_ascii=False,indent=1)
n=len(res); print('nonlidar GT',n)
ok=sum(1 for d in res if d['gor_best'] and d['gor_best']['ok']); print('gor-cand official match any conf:',ok)
for R in (25,50,100,200):
    print(R,'gor cand centroid<R',sum(1 for d in res if d['gor_best'] and d['gor_best']['dist']<R),
          'any nonli<R',sum(1 for d in res if d['any_nonli'] and d['any_nonli']['dist']<R),'any<R',sum(1 for d in res if d['any'] and d['any']['dist']<R))
print('no scene covering GT:',sum(1 for d in res if not d['scenes']))
print('scene kinds',collections.Counter(tuple(sorted(set(x[1] for x in d['scenes']))) for d in res))
