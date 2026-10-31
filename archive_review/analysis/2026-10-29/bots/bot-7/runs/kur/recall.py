"""Recall кандидатов raw/*.npz против эталона курганов Ae (povr+tselye) на нелидарных Ae-участках.
Попадание = официальный _match_quality(круг r=R_C м вокруг кандидата, эталон). Кривая по рангу score внутри сцены (D = кандидатов на км^2)."""
import sys,glob,json,pickle,numpy as np,shapely,os
sys.path.insert(0,'/workspace/expds-koz3'); from expds import metric as M
from shapely.strtree import STRtree
mod=M.load_official('/workspace/expds-koz3/materials/compute_metrics.py'); calc=mod.F1Calculator()
R_C=float(os.environ.get('R_C','2.0'))
S=json.load(open('/workspace/koz3_shared/bot-7/work/gor/scenes.json')); G=pickle.load(open('gt_kur.pkl','rb'))
files=sorted(glob.glob('raw/*.npz')); regs=sorted({S[int(os.path.basename(f)[:3])]['reg'] for f in files})
res={}
for reg in regs:
    sis=[int(os.path.basename(f)[:3]) for f in files if S[int(os.path.basename(f)[:3])]['reg']==reg]
    area=sum(shapely.Polygon(S[i]['corners']).area for i in sis)/1e6
    A=np.concatenate([np.load(f'raw/{i:03d}.npz')['A'] for i in sis])
    # ранг внутри сцены по score -> D_i = ранг/площадь сцены
    Dv=np.zeros(len(A)); off=0
    for i in sis:
        n=len(np.load(f'raw/{i:03d}.npz')['A']); a=shapely.Polygon(S[i]['corners']).area/1e6
        o=np.argsort(-A[off:off+n,2]); r=np.empty(n); r[o]=np.arange(n); Dv[off:off+n]=(r+1)/a; off+=n
    circ=shapely.buffer(shapely.points(A[:,0],A[:,1]),R_C,quad_segs=8)
    tree=STRtree(circ)
    row={'area_km2':round(area,2),'n_cand':len(A)}
    for c in G:
        m=(G[c]['reg']==reg)&(G[c]['mk']=='ae'); gg=shapely.from_wkb(G[c]['wkb'][m])
        best=np.full(len(gg),np.inf); near=np.full(len(gg),np.inf)
        for j,g in enumerate(gg):
            for k in tree.query(g.buffer(5)):
                near[j]=min(near[j],Dv[k])
                ok,_=calc._match_quality(circ[k],g)
                if ok: best[j]=min(best[j],Dv[k])
        row[c]=dict(n=int(m.sum()),**{f'D{d}':int((best<=d).sum()) for d in (50,100,200,400,800,1e9)},near5m_all=int(np.isfinite(near).sum()))
    res[reg]=row; print(reg,row,flush=True)
tot={}
for c in G:
    tot[c]={k:sum(r[c][k] for r in res.values()) for k in res[regs[0]][c]}
print('TOTAL',tot); json.dump(dict(R_C=R_C,per_reg=res,total=tot),open(f'recall_R{R_C}.json','w'),ensure_ascii=False,indent=1)
