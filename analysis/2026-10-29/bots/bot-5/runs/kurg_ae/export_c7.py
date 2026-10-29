"""bot-5. Переэкспорт scores_cand7.json из OOF fam10 (c7g) с полной точностью (в kurg_eval округление до 1e-5 давало связи)."""
import numpy as np, json
OD = '/workspace/koz3_shared/bot-5/analysis/kurg_ae'
D = dict(np.load(f'{OD}/diag_c7g.npz')); P = np.load(f'{OD}/p_oof_fam10_c7g.npy').astype(np.float64)
C7 = dict(np.load('/workspace/koz3_shared/bot-5/tmp/kurg_ae/cand7.npz')); f7 = dict(zip(C7['cand_id'].tolist(), C7['fold'].tolist()))
k2 = {int(D['SID'][k]): k for k in np.flatnonzero(D['SRC'] == 2)}
out = []
for c in C7['cand_id'].tolist():
    k = k2.get(c)
    if k is None or np.isnan(P[k, 0]): out.append(dict(cand_id=c, p=None, p_povr=None, p_tsel=None, fold=f7[c], ok=False)); continue
    out.append(dict(cand_id=c, p=float('%.6g' % (P[k, 1] + P[k, 2])), p_povr=float('%.6g' % P[k, 1]), p_tsel=float('%.6g' % P[k, 2]), fold=f7[c], ok=True))
json.dump(out, open(f'{OD}/scores_cand7.json.tmp', 'w'))
import os; os.replace(f'{OD}/scores_cand7.json.tmp', f'{OD}/scores_cand7.json')
p = np.array([o['p'] for o in out if o['ok']]); print('n', len(out), 'ok', len(p), 'unique p', len(np.unique(p)), 'q50/q99/max', np.quantile(p, [.5, .99]), p.max())
