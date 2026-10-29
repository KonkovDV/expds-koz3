"""bot-7, task 6: пересчёт frag_bot5 по финальному meta.json bot-5 (правило ровно как в export.py:
ближайшая по центру виньетка той же (reg, file), |dx|,|dy| ≤ 192 м, иначе −1). Остальные поля не трогаются
(в export.py от meta.json зависит только frag_bot5). Обновляет D50, _nocentre, _centre_small (json + geojson).
Старые значения сохраняются в comb/frag_old.json."""
import json, collections, hashlib, numpy as np
MP='/workspace/koz3_shared/bot-5/analysis/gor_patch/meta.json'
md5=hashlib.md5(open(MP,'rb').read()).hexdigest()
M5=json.load(open(MP,encoding='utf-8'))
print('meta.json md5',md5,'n',len(M5),flush=True)
by5=collections.defaultdict(list)
for k,m in enumerate(M5): by5[(m['reg'],m['file'])].append((k,m['ux'],m['uy']))
by5={k:(np.array([t[0] for t in v]),np.array([t[1] for t in v]),np.array([t[2] for t in v])) for k,v in by5.items()}
def frag(reg,f,ux,uy):
    t=by5.get((reg,f))
    if t is None: return -1
    K,X,Y=t; d=(X-ux)**2+(Y-uy)**2; q=int(np.argmin(d))   # argmin = первый минимум, как min() в export.py
    return int(K[q]) if abs(X[q]-ux)<=192 and abs(Y[q]-uy)<=192 else -1
R=json.load(open('cands_gor_oof_D50.json'))
old={r['cand_id']:r['frag_bot5'] for r in R}
new={r['cand_id']:frag(r['reg'],r['file'],r['ux'],r['uy']) for r in R}
json.dump(dict(meta_md5_new=md5,old=old),open('comb/frag_old.json','w'))
o=np.array([old[c] for c in old]); n=np.array([new[c] for c in old])
st=dict(n=len(o),filled_old=int((o>=0).sum()),filled_new=int((n>=0).sum()),
        both_same=int(((o>=0)&(o==n)).sum()),both_diff=int(((o>=0)&(n>=0)&(o!=n)).sum()),
        old_only=int(((o>=0)&(n<0)).sum()),new_only=int(((o<0)&(n>=0)).sum()),changed_total=int((o!=n).sum()),
        filled_new_by_kind=dict(collections.Counter(r['kind'] for r in R if new[r['cand_id']]>=0)))
print(json.dumps(st,ensure_ascii=False),flush=True)
json.dump(st,open('comb/refrag_stats.json','w'),ensure_ascii=False,indent=1)
for suf in ('','_nocentre','_centre_small'):
    fj=f'cands_gor_oof_D50{suf}.json'; A=json.load(open(fj)) if suf else R
    for r in A: r['frag_bot5']=new[r['cand_id']]
    json.dump(A,open(fj,'w'),ensure_ascii=False)
    fg=f'cands_gor_oof_D50{suf}.geojson'; Gj=json.load(open(fg))
    for f in Gj['features']: f['properties']['frag_bot5']=new[f['properties']['cand_id']]
    json.dump(Gj,open(fg,'w'),ensure_ascii=False); del Gj
    print(suf or 'full',len(A),'filled',sum(r['frag_bot5']>=0 for r in A),flush=True)
