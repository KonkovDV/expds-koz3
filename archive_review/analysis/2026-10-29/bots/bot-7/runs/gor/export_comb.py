"""bot-7, task 6: экспорт кандидатов с оценкой bot-5 и комбинированным баллом.
Вход: cands_gor_oof_D50{,_nocentre}.json/.geojson (frag_bot5 уже по финальному meta.json, refrag.py), comb/comb_scores_{tag}.npz (comb_eval.py).
Добавляет p_bot5 (OOF p bot-5 или null) и score_comb = honest-10 OOF-выбор w (по AUC на обучающих фолдах; выбран w=1 во всех 10 фолдах и в LOFO):
  оценённые bot-5 и gen != stack_gor_phys -> перцентиль p среди оценённых; остальные -> перцентиль своего score среди всех кандидатов файла.
В geojson confidence = score_comb, сырой балл генератора — в properties.score."""
import json, numpy as np
P={s['cand_id']:s['p'] for s in json.load(open('/workspace/koz3_shared/bot-5/analysis/gor_patch/scores_cand7.json'))}
for suf,tag in (('','D50'),('_nocentre','nocentre')):
    Z=np.load(f'comb/comb_scores_{tag}.npz'); sc=dict(zip(Z['cid'].tolist(),Z['S_h10'].tolist()))
    R=json.load(open(f'cands_gor_oof_D50{suf}.json')); assert len(R)==len(sc)
    for r in R: r['p_bot5']=P.get(r['cand_id']); r['score_comb']=float(sc[r['cand_id']])
    json.dump(R,open(f'cands_gor_oof_D50{suf}_comb.json','w'),ensure_ascii=False)
    Gj=json.load(open(f'cands_gor_oof_D50{suf}.geojson')); assert len(Gj['features'])==len(R)
    sraw={r['cand_id']:r['score'] for r in R}
    for f in Gj['features']:
        q=f['properties']; c=q['cand_id']; q['score']=sraw[c]; q['p_bot5']=P.get(c); q['score_comb']=float(sc[c]); q['confidence']=q['score_comb']
    json.dump(Gj,open(f'cands_gor_oof_D50{suf}_comb.geojson','w'),ensure_ascii=False)
    print(tag,len(R),'with p_bot5',sum(r['p_bot5'] is not None for r in R),'frag_bot5>=0',sum(r['frag_bot5']>=0 for r in R),flush=True)
