"""bot-7, task 6. Экспорт кандидатов городищ в формат FORMAT.md.
OOF (honest-10): для каждого региона вариант генератора выбран по recall на остальных 9 фолдах при бюджете EXPORT_D.
Плюс all-data выбор (для теста). Пишет cands_gor_oof.geojson (EPSG:3857), cands_gor_oof.json (UTM, совместимо с meta.json bot-5)."""
import os, json, collections, numpy as np
from pyproj import Transformer
EXPORT_D=int(os.environ.get('EXPORT_D','50'))
src=open('/workspace/koz3_shared/bot-7/work/gor/cand_eval.py',encoding='utf-8').read()
src=src.split('DS=[')[0]
exec(compile(src,'cand_eval_head','exec'))
D=EXPORT_D
TAB={}
for vi,v in enumerate(VARS):
    if v[0]=='grid': continue
    for i in SC: TAB[(vi,D,i)]=select(i,v,D)
def recall(vi,D,regs,J):
    H=set(); n=0
    for i in SC:
        if S[i]['reg'] not in regs: continue
        h,nn=TAB[(vi,D,i)]; H|=h; n+=nn
    return sum(1 for j in J if greg[j] in regs and j in H),n
VI=[vi for vi,v in enumerate(VARS) if v[0]!='grid']
F10=family_folds(regions,site_family,10)
pick={}
for fi,fo in enumerate(F10):
    tr=set(regions)-set(fo); b=max(VI,key=lambda vi:(recall(vi,D,tr,GTJ)[0],-recall(vi,D,tr,GTJ)[1]))
    for r in fo: pick[r]=(fi,b)
ball=max(VI,key=lambda vi:(recall(vi,D,set(regions),GTJ)[0],-recall(vi,D,set(regions),GTJ)[1]))
print('all-data pick',VARS[ball][0],VARS[ball][2],flush=True)
# фрагменты bot-5
M5=json.load(open('/workspace/koz3_shared/bot-5/analysis/gor_patch/meta.json',encoding='utf-8'))
by5=collections.defaultdict(list)
for k,m in enumerate(M5): by5[(m['reg'],m['file'])].append((k,m['ux'],m['uy']))
feats=[]; rows=[]; cid=0
for i in sorted(SC):
    s=S[i]; reg=s['reg']; fi,vi=pick.get(reg,(-1,ball)); v=VARS[vi]; A=SC[i]['A']
    to3857=Transformer.from_crs(s['epsg'],3857,always_xy=True)
    nsel=max(KMIN,int(round(D*area[i])))
    items=[]
    for rk,q in enumerate(nms_order(i,v)[:nsel] if v[1] else []):
        items.append(dict(gen=GENS[int(A[q,3])],R=float(A[q,4]),score=float(A[q,2]),rank=rk,x=float(A[q,0]),y=float(A[q,1])))
    c=s['poly'].centroid; items.append(dict(gen='centre',R=None,score=1.0,rank=-1,x=c.x,y=c.y))
    hm=SC[i]['hm']
    st=[q for q in srow if T['reg'][q]==reg and s['poly'].contains(sgeo[int(q)].representative_point())]
    for q in st:
        p=sgeo[int(q)].representative_point(); items.append(dict(gen='stack_'+str(T['grp'][q]),R=None,score=float(T['conf'][q]),rank=-1,x=p.x,y=p.y))
    cand5=by5.get((reg,s['file']),[])
    for it in items:
        ux,uy=it['x'],it['y']
        frag=-1
        if cand5:
            k,dx,dy=min(cand5,key=lambda t:(t[1]-ux)**2+(t[2]-uy)**2)
            if abs(dx-ux)<=192 and abs(dy-uy)<=192: frag=k
        box=[[ux-192,uy+192],[ux+192,uy+192],[ux+192,uy-192],[ux-192,uy-192]]
        ctx=[[ux-384,uy+384],[ux+384,uy+384],[ux+384,uy-384],[ux-384,uy-384]]
        row=dict(cand_id=cid,reg=reg,file=s['file'],kind=s['kind'],lidar=bool(s['lidar']),ux=ux,uy=uy,epsg=s['epsg'],
                 corners=box,ctx_corners=ctx,gsd=s['gsd'],gen=it['gen'],R=it['R'],score=it['score'],rank=it['rank'],
                 variant=v[0]+'|'+','.join(map(str,v[2] or ())),fold=fi,frag_bot5=frag)
        rows.append(row)
        X,Y=to3857.transform(ux,uy)
        ang=np.linspace(0,2*np.pi,17)[:-1]
        # круг 10 м в UTM -> 3857
        cx_,cy_=to3857.transform(ux+10*np.cos(ang),uy+10*np.sin(ang))
        ring=[[float(a),float(b)] for a,b in zip(cx_,cy_)]; ring.append(ring[0])
        feats.append(dict(type='Feature',geometry=dict(type='Polygon',coordinates=[ring]),
             properties=dict(class_name='gorodishcha',region=reg,markup_type={'SpOr':'sp_or','Ae':'ae','Or':'or'}[s['kind']],
                             confidence=it['score'],cand_id=cid,gen=it['gen'],rank=it['rank'],fold=fi,frag_bot5=frag,scene_file=s['file'])))
        cid+=1
json.dump(dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name='EPSG:3857')),features=feats),open(f'cands_gor_oof_D{D}.geojson','w'),ensure_ascii=False)
json.dump(rows,open(f'cands_gor_oof_D{D}.json','w'),ensure_ascii=False)
print('written',cid,'candidates; with bot5 frag',sum(1 for r in rows if r['frag_bot5']>=0),collections.Counter(r['gen'].split('_')[0] for r in rows))
