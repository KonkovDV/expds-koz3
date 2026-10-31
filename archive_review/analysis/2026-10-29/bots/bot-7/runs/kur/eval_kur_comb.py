"""bot-7 task7: честный замер новой группы 'ae_kur_b7' (кандидаты курганов на нелидарных Ae) поверх стека.
База = стек (table.pkl, официальные пары _match_quality), вариант = стек + новые кандидаты в классах povr/tselye.
Кривые: официальный жадный проход (как bot-4 gap.py / M.match_collected) по парам; новые пары считаются тем же calc._match_quality.
Критерии как evlib bot-3: honest-10 (FOLDS из base_honest.pkl), LOFO, 5 случайных раскладок, парный бутстрап по семьям, П1–П5.
Прочие классы — счётчики BASE['counts'] (одинаковы в базе и варианте)."""
import sys, os, json, pickle, collections, time
import numpy as np, shapely
from shapely.strtree import STRtree
sys.path.insert(0,'/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0,'/workspace/koz3_shared/bot-10/repo/scripts')
from expds import metric as M
from expds.naming import site_family
from honest_bootstrap import random_family_folds
M._SHM_CACHE_DIR=__import__('pathlib').Path('/nonexistent_ro/x')
mod=M.load_official('/workspace/expds-koz3/materials/compute_metrics.py'); calc=mod.F1Calculator()
t0=time.time()
KC=['kurgany_povrezhdennye','kurgany_tselye']; GRP='ae_kur_b7'; R_C=2.0
D_MAX=float(os.environ.get('D_MAX','400')); TAG=os.environ.get('TAG','')
BASE=pickle.load(open('/workspace/rerank/work/base_honest.pkl','rb')); FOLDS=BASE['folds']; REGIONS=set().union(*FOLDS)
NEW={l.strip() for l in open('/workspace/expds-koz3/notes/regions_las_new.txt',encoding='utf-8') if l.strip()}
OPTICAL=REGIONS-set(json.load(open('/workspace/expds-koz3/notes/las_boxes_59.json',encoding='utf-8')))
LOFO=M.family_folds(REGIONS,site_family,None); RAND=[random_family_folds(REGIONS,10,s) for s in range(5)]
T=pickle.load(open('/workspace/rerank/work/table.pkl','rb'))
G=pickle.load(open('gt_kur.pkl','rb'))
Z=np.load('cands_all.npz',allow_pickle=True); O=np.load('oof.npz')
sel=Z['Dv']<=D_MAX
NA=Z['A'][sel]; Nreg=Z['reg'][sel]; NP={'kurgany_povrezhdennye':O['p_povr'][sel],'kurgany_tselye':O['p_tsel'][sel]}
# --- комбинация с bot-5 (предрегистрация в STATUS.md 29.10): MODE=comb -> 0.5*pct(p_b7)+0.5*pct(p_bot5), pct по всем выбранным кандидатам; p_bot5 нет (ok=false) -> pct_bot5=0; MODE=b5 -> только pct(p_bot5)
MODE=os.environ.get('MODE','comb')
from scipy.stats import rankdata
ids=np.flatnonzero(sel); B5=json.load(open(os.environ['SCORES'])); pos={int(i):k for k,i in enumerate(ids)}
b5={c:np.full(len(ids),np.nan) for c in ('p_povr','p_tsel')}; bad_fold=0; extra=0
for r in B5:
    k=pos.get(int(r['cand_id']))
    if k is None: extra+=1; continue
    if int(r['fold'])!=int(Z['fold'][ids[k]]): bad_fold+=1
    if r.get('ok',True) and r.get('p_povr') is not None: b5['p_povr'][k]=r['p_povr']; b5['p_tsel'][k]=r['p_tsel']
print('bot5 rows',len(B5),'not in ours',extra,'fold mismatch',bad_fold,'scored',int(np.isfinite(b5['p_povr']).sum()),'of',len(ids),flush=True)
assert extra==0 and bad_fold==0
def pct(x):
    f=np.isfinite(x); o=np.zeros(len(x)); o[f]=rankdata(x[f])/f.sum(); return o
for c,kk in (('kurgany_povrezhdennye','p_povr'),('kurgany_tselye','p_tsel')):
    NP[c]=0.5*pct(NP[c])+0.5*pct(b5[kk]) if MODE=='comb' else pct(b5[kk])
VARSEL=os.environ.get('VARS','both').split(',')
print('new cands',int(sel.sum()),'D_MAX',D_MAX,flush=True)
circ=shapely.buffer(shapely.points(NA[:,0],NA[:,1]),R_C,quad_segs=8)
stk={}; newpairs={}
for c in KC:
    k=T['CL'].index(c); idx=np.flatnonzero(T['cls']==k)
    P,J,S=T['pairs'][(c,c)]
    stk[c]=dict(idx=idx,conf=T['conf'][idx].astype(np.float64),reg=T['reg'][idx].astype(str),grp=T['grp'][idx].astype(str),P=P,J=J,S=S)
    gg=shapely.from_wkb(G[c]['wkb']); tree=STRtree([g.buffer(calc.r_tolerance) if g.geom_type=='Point' else g for g in gg])
    qi,qj=tree.query(circ); pp=[]
    for a,b in zip(qi,qj):
        ok,s=calc._match_quality(circ[a],gg[b])
        if ok: pp.append((int(a),int(b),float(s)))
    newpairs[c]=pp; print(c,'stack',len(idx),'new pairs',len(pp),f'{time.time()-t0:.0f}s',flush=True)
del T
def curve(c,with_new,scope,scale=1.0,new_mask=None):
    s=stk[c]; greg=G[c]['reg']
    pos={int(g):i for i,g in enumerate(s['idx'])}
    by=collections.defaultdict(list)
    for a,b,q in zip(s['P'],s['J'],s['S']):
        i=pos.get(int(a))
        if i is not None: by[i].append((int(b),float(q)))
    conf=s['conf']; reg=s['reg']; grp=s['grp']; n0=len(conf)
    if with_new:
        m=np.ones(len(NA),bool) if new_mask is None else new_mask
        keep=np.flatnonzero(m); remap={int(k):n0+t for t,k in enumerate(keep)}
        for a,b,q in newpairs[c]:
            if a in remap: by[remap[a]].append((b,q))
        conf=np.r_[conf,NP[c][keep]*scale]; reg=np.r_[reg,Nreg[keep]]; grp=np.r_[grp,np.full(len(keep),GRP)]
    o=np.argsort(-conf,kind='stable'); used=set(); hit=np.zeros(len(o),bool)
    for r,i in enumerate(o):
        lst=by.get(int(i))
        if not lst: continue
        best,bj=-np.inf,-1
        for j,q in sorted(lst):
            if j in used or (scope=='site' and greg[j]!=reg[i]): continue
            if q>best+1e-9: best,bj=q,j
        if bj>=0: used.add(bj); hit[r]=True
    return M.ClassCurve(conf=conf[o],hit=hit,pred_region=list(reg[o]),n_gt=len(greg),gt_regions=collections.Counter(greg),pred_group=list(grp[o]))
def counts_with(cv,folds,unseen):
    cs,info=M.honest_counts(cv,folds,unseen=unseen); full=dict(BASE['counts']); full.update(cs); return full,info
def sc(cnt): return [M.score_counts(cnt)[0],M.score_counts(cnt,NEW)[0],M.score_counts(cnt,OPTICAL)[0]]
def boot95(a,b,n=2000,seed=0):
    # копия M.paired_bootstrap_counts (те же семья/seed -> те же выборки), добавлены перцентили 2.5/97.5
    regs={r for cs in (a,b) for by in cs.values() for r in by}; fams={}
    for r in sorted(regs): fams.setdefault(site_family(r),[]).append(r)
    names=sorted(fams); rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        pick=np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names))
        w={r:float(k) for f,k in zip(names,pick) if k for r in fams[f]}
        v.append(M.score_counts(a,None,w)[0]-M.score_counts(b,None,w)[0])
    v=np.asarray(v); return {'p2_5':float(np.percentile(v,2.5)),'p97_5':float(np.percentile(v,97.5))}
def criteria(cA,cB,unseen):
    a,ia=counts_with(cA,FOLDS,unseen); b,ib=counts_with(cB,FOLDS,unseen); sa,sb=sc(a),sc(b); d=sa[0]-sb[0]
    pb=M.paired_bootstrap_counts(a,b,site_family,n=2000,seed=0); pb.update(boot95(a,b))
    lofo=sc(counts_with(cA,LOFO,unseen)[0])[0]-sc(counts_with(cB,LOFO,unseen)[0])[0]
    rnd=[sc(counts_with(cA,f,unseen)[0])[0]-sc(counts_with(cB,f,unseen)[0])[0] for f in RAND]
    pa,pbb=M.score_counts(a)[1],M.score_counts(b)[1]; den=sum(M.CLASS_WEIGHTS[c] for c in pa if pa[c]['n_gt']>0)
    wd={c:M.CLASS_WEIGHTS[c]*(pa[c]['f1']-pbb[c]['f1'])/den for c in pa}
    # вклад нелидарных Ae-участков с кандидатами
    R=set(Nreg.tolist()); loc={c:{k:[sum(a[c].get(r,[0,0,0])[i] for r in R) for i in range(3)] for k,a in (('var',a),('base',b))} for c in KC}
    res=dict(base=sb,var=sa,d_honest10=d,boot=pb,d_lofo=lofo,d_rand=rnd,w_delta=wd,
             per_class_var={c:pa[c] for c in KC},per_class_base={c:pbb[c] for c in KC},tp_fp_gt_on_ae_regions=loc,unseen_info=(ia,ib))
    res['P1']=bool(d>=0.002); res['P2']=bool(pb.get('p_gt0',0)>=0.85); res['P3']=bool(lofo>0 and sum(x>0 for x in rnd)>=4)
    res['P4']=bool(all(wd[c]>=-0.0005 for c in ['kurgany_povrezhdennye','kurgany_tselye','gorodishcha']) and sa[1]>=sb[1]-0.001 and sa[2]>=sb[2]-0.001)
    res['P5_below_MDE']=bool(d<0.006); return res
OUT={}
SCOPES=os.environ.get('SCOPES','global,site').split(','); UNS=os.environ.get('UNSEEN','drop,keep').split(',')
VARS_ENV=os.environ.get('VARS')
for scope in SCOPES:
    cb={c:curve(c,False,scope) for c in KC}
    if scope=='global':
        cnt,_=counts_with(cb,FOLDS,'drop'); OUT['repro_base_honest10']=M.score_counts(cnt)[0]
        OUT['repro_match_BASEcounts']={c:cnt[c]==BASE['counts'][c] for c in KC}
        print('repro base',OUT['repro_base_honest10'],OUT['repro_match_BASEcounts'],flush=True)
    VAR={'both':dict(scale=1.0,cls=KC),'povr':dict(scale=1.0,cls=KC[:1]),'tsel':dict(scale=1.0,cls=KC[1:]),'both_stackfirst':dict(scale=1e-3,cls=KC)}
    for vn,v in VAR.items():
        if VARS_ENV and vn not in VARS_ENV.split(','): continue
        if vn not in VARSEL: continue
        cv=None
        cv={c:(curve(c,True,scope,v['scale']) if c in v['cls'] else cb[c]) for c in KC}
        for unseen in UNS:
            r=criteria(cv,cb,unseen); OUT[f'{scope}|{vn}|{unseen}']=r
            print(scope,vn,unseen,'d10=%.5f lofo=%.5f rnd=%s P=%.3f CI90=[%.5f,%.5f] CI95=[%.5f,%.5f] P1-4=%s%s%s%s wd=%s'%(r['d_honest10'],r['d_lofo'],[round(x,5) for x in r['d_rand']],
                  r['boot']['p_gt0'],r['boot']['p5'],r['boot']['p95'],r['boot']['p2_5'],r['boot']['p97_5'],r['P1'],r['P2'],r['P3'],r['P4'],{k[:10]:round(x,5) for k,x in r['w_delta'].items() if abs(x)>1e-9}),f'{time.time()-t0:.0f}s',flush=True)
json.dump(OUT,open(f'eval_kur_{MODE}{TAG}_D{int(D_MAX)}_{"-".join(SCOPES)}_{VARS_ENV}.json','w'),ensure_ascii=False,indent=1,default=lambda x: x if not hasattr(x,'tolist') else x.tolist())
print('done',flush=True)
