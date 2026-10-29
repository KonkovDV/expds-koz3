"""bot-7 task7: свод кандидатов raw/*.npz -> cands_all.npz (признаки + соседство + метки + фолд) и экспорт для bot-5.
Метка класса c = официальный _match_quality(круг R_C=2 м, любой эталон класса c того же участка) — как пройдёт в метрике.
fold = индекс в FOLDS из /workspace/rerank/work/base_honest.pkl (honest-10 стека; те же фолды, что в замере)."""
import sys,glob,json,pickle,os,numpy as np,shapely
sys.path.insert(0,'/workspace/expds-koz3'); from expds import metric as M
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
mod=M.load_official('/workspace/expds-koz3/materials/compute_metrics.py'); calc=mod.F1Calculator()
R_C=2.0; EXPORT_D=float(os.environ.get('EXPORT_D','400'))
S=json.load(open('/workspace/koz3_shared/bot-7/work/gor/scenes.json')); G=pickle.load(open('gt_kur.pkl','rb'))
FOLDS=pickle.load(open('/workspace/rerank/work/base_honest.pkl','rb'))['folds']
fold_of={r:k for k,fo in enumerate(FOLDS) for r in fo}
A=[];F=[];FN=None;Dv=[];Rk=[]
for f in sorted(glob.glob('raw/*.npz')):
    z=np.load(f); a=z['A']; FN=list(z['FN']); si=int(os.path.basename(f)[:3])
    ar=shapely.Polygon(S[si]['corners']).area/1e6
    o=np.argsort(-a[:,2],kind='stable'); r=np.empty(len(a)); r[o]=np.arange(len(a))
    A.append(a); F.append(z['F']); Dv.append((r+1)/ar); Rk.append(r.astype(int))
A=np.concatenate(A); F=np.concatenate(F); Dv=np.concatenate(Dv); Rk=np.concatenate(Rk)
si=A[:,4].astype(int); reg=np.array([S[i]['reg'] for i in si]); gsd=np.array([S[i]['gsd'] for i in si])
# соседство (без меток): число кандидатов и сильных кандидатов (D<=100) в 15/40/100 м
ex=[];exn=[]
for i in np.unique(si):
    m=np.flatnonzero(si==i); t=cKDTree(A[m,:2]); strong=Dv[m]<=100; ts=cKDTree(A[m][strong,:2]) if strong.any() else None
    cols=[]
    for rr in (15,40,100):
        cols.append(np.array([len(x) for x in t.query_ball_point(A[m,:2],rr)])-1)
        cols.append(np.array([len(x) for x in ts.query_ball_point(A[m,:2],rr)]) if ts is not None else np.zeros(len(m)))
    e=np.zeros((len(m),6)); e[:]=np.c_[tuple(cols)]; ex.append((m,e))
NB=np.zeros((len(A),6))
for m,e in ex: NB[m]=e
FN=FN+['nb15','nbs15','nb40','nbs40','nb100','nbs100','gsd','logD','score']
X=np.c_[F,NB,gsd,np.log(Dv),A[:,2]].astype(np.float32)
circ=shapely.buffer(shapely.points(A[:,0],A[:,1]),R_C,quad_segs=8); tree=STRtree(circ)
Y={}
for c in G:
    y=np.zeros(len(A),np.int8); gg=shapely.from_wkb(G[c]['wkb']); greg=G[c]['reg']; mk=G[c]['mk']
    for j in np.flatnonzero(np.isin(greg,np.unique(reg))):
        for k in tree.query(gg[j]):
            if reg[k]==greg[j] and not y[k]:
                ok,_=calc._match_quality(circ[k],gg[j])
                if ok: y[k]=1 if mk[j]=='ae' else 2
    Y[c]=y; print(c,'pos ae',int((y==1).sum()),'pos other markup',int((y==2).sum()),flush=True)
fold=np.array([fold_of.get(r,-1) for r in reg]); assert (fold>=0).all()
np.savez('cands_all.npz',A=A,X=X,FN=np.array(FN),Dv=Dv,rank=Rk,reg=reg,si=si,fold=fold,y_povr=Y['kurgany_povrezhdennye'],y_tsel=Y['kurgany_tselye'])
sel=np.flatnonzero(Dv<=EXPORT_D)
GN=['log2','log3','log4.5','log6.5','log9','log13']
recs=[dict(cand_id=int(k),reg=str(reg[k]),file=S[si[k]]['file'],ux=float(A[k,0]),uy=float(A[k,1]),epsg=int(S[si[k]]['epsg']),gsd=float(gsd[k]),
           gen=GN[int(A[k,3])],score=float(A[k,2]),rank=int(Rk[k]),D=float(Dv[k]),fold=int(fold[k])) for k in sel]
json.dump(recs,open('cands_kur_ae.json','w'),ensure_ascii=False)
print('total',len(A),'exported',len(recs),'EXPORT_D',EXPORT_D)
