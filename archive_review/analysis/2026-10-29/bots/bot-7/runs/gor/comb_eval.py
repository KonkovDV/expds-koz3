"""bot-7, task 6: честная оценка ранжирования кандидатов городищ: свой балл vs p bot-5 vs комбинация.
Метка кандидата = офиц. совпадение круга 10 м с эталоном городища (_match_quality, как verify_export.py).
Шкалы (без меток, заранее): u_own = перцентиль сырого score среди всех кандидатов файла; u_p = перцентиль p среди оценённых.
Комбинация (только оценённые bot-5 и gen != stack_gor_phys): comb_w = w*u_p + (1-w)*u_own; иначе comb = u_own.
  fixed: w=0.5 (ранговое среднее, заранее), w=1 (p где есть); sel: w из {0,.25,.5,.75,1} выбирается на обучающих фолдах
  по AUC кандидатов (метка выше) — honest-10 (поле fold) и LOFO (family_folds(...,None)).
p bot-5 — OOF по нашему полю fold, семья кандидата исключена из обучения (проверено в comb/score_cand7_repro.py).
Метрики: AUC кандидатов; recall@B (глобальный топ-B) и recall@k на сцену (топ-k в каждой сцене) по эталонам:
73 без лидара и все 118 (в сценах 99). Бутстреп по семьям (1000) для разностей comb−own."""
import json, collections, sys, os, numpy as np
from common import *
from scipy.stats import rankdata
from expds.metric import family_folds
from expds.naming import site_family
T,G,greg,gmk,lid,mod=load_gt(); calc=mod.F1Calculator()
regions=sorted(set(T['reg'])|set(T['greg'][C])); del T
CF=sys.argv[1] if len(sys.argv)>1 else 'cands_gor_oof_D50.json'
PF=os.environ.get('PFILE','/workspace/koz3_shared/bot-5/analysis/gor_patch/scores_cand7.json')
R=json.load(open(CF)); N=len(R)
P={s['cand_id']:s for s in json.load(open(PF))}
cid=np.array([r['cand_id'] for r in R]); gen=np.array([r['gen'] for r in R]); reg=np.array([r['reg'] for r in R])
scene=np.array([r['reg']+'|'+r['file'] for r in R]); fold=np.array([r['fold'] for r in R]); kind=np.array([r['kind'] for r in R])
raw=np.array([r['score'] for r in R],float); lidr=np.array([r['lidar'] for r in R])
p=np.array([P[c]['p'] if c in P else np.nan for c in cid]); scored=~np.isnan(p)
use_p=scored&(gen!='stack_gor_phys')
# --- попадания
hits=[[] for _ in range(N)]
byr=collections.defaultdict(list)
for i,r in enumerate(R): byr[r['reg']].append(i)
for j in range(len(G)):
    ii=byr.get(greg[j],[]); 
    if not ii: continue
    x0,y0,x1,y1=G[j].buffer(12).bounds
    for i in ii:
        ux,uy=R[i]['ux'],R[i]['uy']
        if x0<=ux<=x1 and y0<=uy<=y1 and calc._match_quality(Point(ux,uy).buffer(10),G[j])[0]: hits[i].append(j)
y=np.array([len(h)>0 for h in hits])
NL=set(np.flatnonzero(~lid)); ALLJ=set(range(len(G)))
cover_nl=len({j for h in hits for j in h if j in NL}); cover_all=len({j for h in hits for j in h})
print('cands',N,'pos',int(y.sum()),'scored',int(scored.sum()),'use_p',int(use_p.sum()),'GT covered nonlidar',cover_nl,'/',len(NL),'all',cover_all,'/',len(G),flush=True)
pct=lambda v:(rankdata(v)-0.5)/len(v)
u_own=pct(raw); u_p=np.full(N,np.nan); u_p[scored]=pct(p[scored])
def comb(w):
    s=u_own.copy(); s[use_p]=w*u_p[use_p]+(1-w)*u_own[use_p]; return s
def auc(s,m):
    yy=y[m]; ss=s[m]; n1=yy.sum(); n0=len(yy)-n1
    if n1==0 or n0==0: return float('nan')
    rk=rankdata(ss); return float((rk[yy].sum()-n1*(n1+1)/2)/(n1*n0))
WS=[0,0.25,0.5,0.75,1.0]
fam=np.array([site_family(r) for r in reg])
LOFO=family_folds(regions,site_family,None)
lofo_of={r:k for k,fo in enumerate(LOFO) for r in fo}; lfold=np.array([lofo_of[r] for r in reg])
def oof_sel(foldarr):
    s=np.zeros(N); picks=collections.Counter()
    for f in np.unique(foldarr):
        te=foldarr==f; tr=~te
        w=max(WS,key=lambda w:(round(auc(comb(w),tr),6),-abs(w-0.5)))  # ничья -> ближе к 0.5
        picks[w]+=1; s[te]=comb(w)[te]
    return s,dict(picks)
S_h10,pk10=oof_sel(fold); S_lofo,pklofo=oof_sel(lfold)
SC={'own':u_own,'p_only_where_scored':comb(1.0),'rankavg_w0.5':comb(0.5),'sel_honest10':S_h10,'sel_LOFO':S_lofo}
print('picks honest10',pk10,'LOFO',pklofo,flush=True)
sc_idx=collections.defaultdict(list)
for i in range(N): sc_idx[scene[i]].append(i)
def cover(sel,J): return len({j for i in sel for j in hits[i] if j in J})
def rec_global(s,m,B): 
    ii=np.flatnonzero(m); ii=ii[np.argsort(-s[ii],kind='stable')][:B]; return cover(ii,NL),cover(ii,ALLJ)
def rec_scene(s,m,k):
    sel=[]
    for sc_,ii in sc_idx.items():
        ii=np.array([i for i in ii if m[i]])
        if len(ii): sel+=list(ii[np.argsort(-s[ii],kind='stable')][:k])
    return cover(sel,NL),cover(sel,ALLJ),len(sel)
BS=[100,250,500,1000,2000,5000,10000]; KS=[1,2,3,5,10]
def report(s,m):
    return dict(auc_all=auc(s,m),auc_nonlidar=auc(s,m&~lidr),
                rec_global={B:rec_global(s,m,B) for B in BS},rec_scene={k:rec_scene(s,m,k) for k in KS})
out=dict(cands=CF,pfile=PF,n=N,pos=int(y.sum()),n_scored=int(scored.sum()),n_use_p=int(use_p.sum()),
         gt_cover_nonlidar=cover_nl,gt_cover_all=cover_all,picks_honest10=pk10,picks_LOFO=pklofo,full={},scored={})
allm=np.ones(N,bool)
for nm,s in SC.items(): out['full'][nm]=report(s,allm); print('FULL',nm,json.dumps(out['full'][nm]),flush=True)
# оценённое подмножество: own vs p vs comb
SCs={'own':u_own,'p':np.where(scored,p,-1.0),'rankavg_w0.5':comb(0.5),'sel_honest10':S_h10,'sel_LOFO':S_lofo}
out['scored_cover']=dict(nonlidar=cover(np.flatnonzero(scored),NL),all=cover(np.flatnonzero(scored),ALLJ),pos=int(y[scored].sum()))
for nm,s in SCs.items(): out['scored'][nm]=report(s,scored); print('SCORED',nm,json.dumps(out['scored'][nm]),flush=True)
# по генераторам на оценённых (сверка с bot-5)
out['by_gen_scored']={g:dict(n=int((scored&(gen==g)).sum()),pos=int(y[scored&(gen==g)].sum()),auc_p=auc(np.nan_to_num(p,nan=-1),scored&(gen==g)),auc_own=auc(raw,scored&(gen==g))) for g in np.unique(gen[scored])}
print('by gen',out['by_gen_scored'],flush=True)
# бутстреп по семьям: разность comb−own (AUC без лидара; recall@3/сцену без лидара)
rng=np.random.default_rng(0); F=np.unique(fam); fi={f:np.flatnonzero(fam==f) for f in F}
gfam=np.array([site_family(r) for r in greg])
def boot(sa,sb,m,nrep=1000):
    d1=[];d2=[]
    base_sel={}
    for s_,key in ((sa,'a'),(sb,'b')):
        sel=set()
        for sc_,ii in sc_idx.items():
            ii=np.array([i for i in ii if m[i]])
            if len(ii): sel|=set(ii[np.argsort(-s_[ii],kind='stable')][:3].tolist())
        base_sel[key]={j for i in sel for j in hits[i] if j in NL}
    for _ in range(nrep):
        ff=rng.choice(F,len(F)); idx=np.concatenate([fi[f] for f in ff])
        yy=y[idx]; mm=m[idx]&~lidr[idx]
        def a(s_):
            ss=s_[idx][mm]; t=yy[mm]; n1=t.sum(); n0=len(t)-n1
            return (rankdata(ss)[t].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan
        d1.append(a(sb)-a(sa))
        cnt=collections.Counter(ff); 
        d2.append(sum(cnt[gfam[j]] for j in base_sel['b'])-sum(cnt[gfam[j]] for j in base_sel['a']))
    q=lambda d:[float(np.nanpercentile(d,2.5)),float(np.nanmedian(d)),float(np.nanpercentile(d,97.5)),float(np.nanmean(np.array(d)>0))]
    return dict(d_auc_nonlidar=q(d1),d_rec3scene_nonlidar=q(d2))
out['boot']={}
for nm in ('rankavg_w0.5','p_only_where_scored','sel_honest10','sel_LOFO'):
    out['boot'][nm]=boot(u_own,SC[nm],allm); print('BOOT',nm,out['boot'][nm],flush=True)
tag=os.environ.get('TAG','D50')
json.dump(out,open(f'comb/comb_eval_{tag}.json','w'),ensure_ascii=False,indent=1,default=float)
np.savez(f'comb/comb_scores_{tag}.npz',cid=cid,u_own=u_own,p=p,y=y,S_h10=S_h10,S_lofo=S_lofo,comb05=comb(0.5))
