"""bot-7, task 6. Recall кандидатов городищ (офиц. _match_quality, кандидат = круг 10 м вокруг точки).
Честно: вариант генератора выбирается по recall на обучающих фолдах (honest-10 family_folds / LOFO / 5 случайных раскладок),
применяется к отложенному. Бюджет D = кандидатов на км^2 сцены (минимум KMIN на сцену), задан сеткой заранее."""
import json, collections, pickle, os, sys, numpy as np, shapely
from shapely.geometry import Point
from shapely.strtree import STRtree
from common import *
sys.path.insert(0,'/workspace/koz3_shared/bot-10/repo/scripts')
from expds.metric import family_folds
from expds.naming import site_family
T,G,greg,gmk,lid,mod=load_gt(); calc=mod.F1Calculator(); S=load_scenes()
GENS=['frst_abs','frst_bright','frst_dark','log_bright','log_dark','grad']; RADII=[15,25,40,60,90,140]
regions=sorted(set(T['reg'])|set(T['greg'][C]))
# --- сцены и кандидаты
done=[i for i in range(len(S)) if os.path.exists(f'raw/{i:03d}.npz')]
area={i:S[i]['poly'].area/1e6 for i in done}
# GT -> сцены (только обработанные)
gt_sc=collections.defaultdict(list)
for j in range(len(G)):
    for i in done:
        if S[i]['reg']==greg[j] and S[i]['poly'].intersects(G[j]): gt_sc[j].append(i)
GTJ=sorted(gt_sc); print('GT in processed scenes',len(GTJ),'of',len(G),'nonlidar',sum(1 for j in GTJ if not lid[j]),flush=True)
TREE=STRtree([G[j].buffer(12) for j in range(len(G))])
def hits_of(xy,reg):
    """для каждой точки: список GT j, которые офиц. совпадают с кругом 10 м"""
    if len(xy)==0: return {}
    pts=shapely.points(xy[:,0],xy[:,1])
    pi,gi=TREE.query(pts,predicate='intersects')
    hm=collections.defaultdict(list)
    for a,b in zip(pi,gi):
        if greg[b]!=reg: continue
        if calc._match_quality(pts[a].buffer(10),G[b])[0]: hm[int(a)].append(int(b))
    return hm
# raw по сцене: ранги по каждому варианту; кэш попаданий
k_stack=T['CL'].index(C); srow=np.flatnonzero((T['cls']==k_stack)&np.isin(T['grp'],['sp_or','ae','or','gor_phys']))
W=pickle.load(open('/workspace/rerank/work/geom_wkb.pkl','rb')); sgeo={int(i):shapely.from_wkb(W[i]) for i in srow}; del W
SC={}
for i in done:
    A=np.load(f'raw/{i:03d}.npz')['A']; s=S[i]
    c=s['poly'].centroid
    st=[i_ for i_ in srow if T['reg'][i_]==s['reg'] and s['poly'].contains(sgeo[int(i_)].representative_point())]
    stxy=np.array([[sgeo[int(q)].representative_point().x,sgeo[int(q)].representative_point().y] for q in st]).reshape(-1,2)
    stc=T['conf'][st] if st else np.zeros(0)
    allxy=np.r_[A[:,:2],[[c.x,c.y]],stxy]
    hm=hits_of(allxy,s['reg'])
    SC[i]=dict(A=A,n=len(A),stn=len(st),stc=stc,hm=hm)
print('scenes',len(SC),flush=True)
def variant_list():
    V=[]
    for g in range(len(GENS)):
        for R in RADII: V.append((GENS[g],(g,),(R,)))
        V.append((GENS[g]+'@all',(g,),tuple(RADII)))
    V.append(('frst_abs+log_bright@all',(0,3),tuple(RADII)))
    V.append(('all@all',tuple(range(6)),tuple(RADII)))
    V.append(('grid',None,None))
    V.append(('none',(),()))
    return V
VARS=variant_list()
KMIN=3
MODE=os.environ.get('MODE','D')  # D: кандидатов на км^2 (мин. KMIN); K: ровно K на сцену
ORD={}
def nms_order(i,v):
    key=(i,v[0],v[2])
    if key in ORD: return ORD[key]
    A=SC[i]['A']; name,gs,Rs=v
    m=np.isin(A[:,3],gs)&np.isin(A[:,4],Rs); ii=np.flatnonzero(m); ii=ii[np.argsort(-A[ii,2],kind='stable')]
    cell={}; keep=[]
    for q in ii:
        x,y=A[q,0],A[q,1]; cx_,cy_=int(x//40),int(y//40); ok=True
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for (px,py) in cell.get((cx_+dx,cy_+dy),()):
                    if (px-x)**2+(py-y)**2<1600: ok=False; break
                if not ok: break
            if not ok: break
        if ok: keep.append(int(q)); cell.setdefault((cx_,cy_),[]).append((x,y))
    ORD[key]=keep; return keep
def select(i,v,D,with_centre=True,with_stack=True):
    s=SC[i]; name,gs,Rs=v; nsel=D if MODE=='K' else max(KMIN,int(round(D*area[i])))
    idx=nms_order(i,v)[:nsel] if gs else []
    hit=set()
    for q in idx: hit.update(s['hm'].get(int(q),[]))
    n=len(idx)
    if with_centre: hit.update(s['hm'].get(s['n'],[])); n+=1
    if with_stack:
        for t in range(s['stn']): hit.update(s['hm'].get(s['n']+1+t,[]))
        n+=s['stn']
    return hit,n
def grid_hits(i,D):
    """базовая линия: регулярная сетка с шагом 1/sqrt(D) км внутри сцены"""
    s=S[i]; P=s['poly']; step=1000/np.sqrt(D if MODE=='D' else max(D/area[i],1e-3)); x0,y0,x1,y1=P.bounds
    xs,ys=np.meshgrid(np.arange(x0+step/2,x1,step),np.arange(y0+step/2,y1,step)); xy=np.c_[xs.ravel(),ys.ravel()]
    xy=xy[shapely.contains_xy(P,xy[:,0],xy[:,1])]
    hm=hits_of(xy,s['reg']); h=set(); [h.update(v) for v in hm.values()]; return h,len(xy)
DS=[1,3,5,10,20,50] if MODE=='K' else [5,10,20,50,100,200]
# предвычисление: для каждого варианта, D, сцены -> (hits, n)
TAB={}
for vi,v in enumerate(VARS):
    for D in DS:
        for i in SC:
            TAB[(vi,D,i)]=grid_hits(i,D) if v[0]=='grid' else select(i,v,D)
print('tab done',flush=True)
def recall(vi,D,regs,J):
    H=set(); n=0; a=0
    for i in SC:
        if S[i]['reg'] not in regs: continue
        h,nn=TAB[(vi,D,i)]; H|=h; n+=nn; a+=area[i]
    Jr=[j for j in J if greg[j] in regs]
    return sum(1 for j in Jr if j in H),len(Jr),n,a
JALL=GTJ; JNL=[j for j in GTJ if not lid[j]]
def honest(folds,D,score_set):
    """для каждого фолда выбрать вариант по train-recall на score_set, применить к отложенному"""
    hit=0; tot=0; n=0; a=0; hitnl=0; totnl=0; picks=collections.Counter()
    for fo in folds:
        tr=set(regions)-set(fo)
        best=max(range(len(VARS)),key=lambda vi:(recall(vi,D,tr,score_set)[0],-recall(vi,D,tr,score_set)[2]))
        picks[VARS[best][0]]+=1
        h,t,nn,aa=recall(best,D,set(fo),JALL); hit+=h; tot+=t; n+=nn; a+=aa
        h2,t2,_,_=recall(best,D,set(fo),JNL); hitnl+=h2; totnl+=t2
    return dict(recall_all=f'{hit}/{tot}',recall_nonlidar=f'{hitnl}/{totnl}',n_cand=n,km2=round(a,1),per_km2=round(n/a,1),picks=dict(picks))
F10=family_folds(regions,site_family,10); LOFO=family_folds(regions,site_family,None)
from honest_bootstrap import random_family_folds
SEEDS=[random_family_folds(regions,10,s) for s in range(5)]
out={'n_scenes':len(SC),'scene_km2':round(sum(area.values()),1),'n_scenes_nonlidar':sum(1 for i in SC if not S[i]['lidar']),
     'km2_nonlidar':round(sum(area[i] for i in SC if not S[i]['lidar']),1),'gt_all':len(JALL),'gt_nonlidar':len(JNL),'by_budget':{},'fixed':{}}
# фиксированные (без выбора) для справки: centre+stack only, grid
for D in DS:
    vn=[vi for vi,v in enumerate(VARS) if v[0]=='none'][0]; vg=[vi for vi,v in enumerate(VARS) if v[0]=='grid'][0]
    h,t,n,a=recall(vn,D,set(regions),JALL); h2,t2,_,_=recall(vn,D,set(regions),JNL)
    g,tg,ng,ag=recall(vg,D,set(regions),JALL); g2,_,_,_=recall(vg,D,set(regions),JNL)
    out['fixed'][D]={'centre+stack':dict(all=f'{h}/{t}',nonlidar=f'{h2}/{t2}',n=n),'grid_only':dict(all=f'{g}/{tg}',nonlidar=f'{g2}/{t2}',n=ng)}
    # без стека/центра (чистые генераторы, all-data, для справки)
    best={}
    for vi,v in enumerate(VARS):
        if v[0] in ('grid','none'): continue
        H=set(); n=0
        for i in SC:
            hh,nn=select(i,v,D,False,False); H|=hh; n+=nn
        best[v[0]+'|'+','.join(map(str,v[2]))]=(sum(1 for j in JNL if j in H),sum(1 for j in JALL if j in H),n)
    out['fixed'][D]['pure_top5_nonlidar']=sorted(best.items(),key=lambda x:-x[1][0])[:5]
    r={}
    for nm,fs in (('honest10',F10),('lofo',LOFO)): r[nm]=honest(fs,D,JALL)
    r['seeds_nonlidar']=[honest(fs,D,JALL)['recall_nonlidar'] for fs in SEEDS]
    out['by_budget'][D]=r; print(D,json.dumps(r,ensure_ascii=False),flush=True); print('  fixed',json.dumps(out['fixed'][D],ensure_ascii=False),flush=True)
vn=[vi for vi,v in enumerate(VARS) if v[0]=='none'][0]
dec={}
for nm,(wc,ws) in {'centre_only':(True,False),'stack_only':(False,True)}.items():
    H=set(); n=0
    for i in SC: h,nn=select(i,VARS[vn],5,wc,ws); H|=h; n+=nn
    dec[nm]=dict(all=f"{sum(1 for j in JALL if j in H)}/{len(JALL)}",nonlidar=f"{sum(1 for j in JNL if j in H)}/{len(JNL)}",n=n)
out['decomp']=dec
vol={}
for D in DS:
    r={}
    for nm,fs in (('honest10',F10),):
        per=collections.defaultdict(lambda:[0,0,0.0])
        for fo in fs:
            tr=set(regions)-set(fo)
            best=max(range(len(VARS)),key=lambda vi:(recall(vi,D,tr,JALL)[0],-recall(vi,D,tr,JALL)[2]))
            for i in SC:
                if S[i]['reg'] in fo:
                    k=(S[i]['kind'],'li' if S[i]['lidar'] else 'noli'); per[k][0]+=1; per[k][1]+=TAB[(best,D,i)][1]; per[k][2]+=area[i]
        r={f'{a}|{b}':dict(scenes=v[0],cand=v[1],km2=round(v[2],1),per_scene=round(v[1]/v[0],1),per_km2=round(v[1]/max(v[2],1e-9),1)) for (a,b),v in per.items()}
    vol[D]=r
out['volume_honest10']=vol; out['mode']=MODE
print('decomp',dec); print('volume',json.dumps(vol,ensure_ascii=False))
json.dump(out,open(f'res_cand_{MODE}.json','w'),ensure_ascii=False,indent=1)
