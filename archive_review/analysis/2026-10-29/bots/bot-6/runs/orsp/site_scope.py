"""Site-scope SPF (bot-6): пары кандидат–эталон разных участков выброшены (≈ EXPDS_MATCH_SCOPE=site), canon honest-10, A vs SPF."""
import sys, json, numpy as np
import time, pickle
from pathlib import Path
from collections import Counter
from scipy.spatial import cKDTree
sys.path.insert(0, '/workspace/rerank'); sys.path.insert(0, '/workspace/expds-koz3')
import fast
from common import site_family, family_folds, score_counts, paired_bootstrap_counts, honest_counts
T = fast.T
for (pc, gc), d in fast.PL.items():
    gr = T['greg'][gc]
    for p in list(d):
        d[p] = [(j, s) for j, s in d[p] if gr[j] == T['reg'][p]]
        if not d[p]: del d[p]
from common import sets, load, W
from expds.metric import CLASS_WEIGHTS as WT
from scipy.spatial import cKDTree
OUT = Path(__file__).parent; t0 = time.time()
T = fast.T; CL = fast.CL; cls = T['cls'].astype(int); conf = T['conf']; reg = T['reg']; grp = T['grp']; cx, cy = T['cx'], T['cy']
gt = load(W / 'gt_train_v2.geojson'); new, optical = sets(gt); del gt
K0, K1 = fast.cid['kurgany_povrezhdennye'], fast.cid['kurgany_tselye']
lat = np.arctan(np.sinh(cy / 6378137.0)); gx = cx * np.cos(lat); gy = cy * np.cos(lat)
spm = ((cls == K0) | (cls == K1)) & np.isin(grp, ['sp_focal', 'sp_focal2'])
o_any = np.zeros(len(cls)); o_same = np.zeros(len(cls)); nconf = Counter()
for r in sorted(set(reg[spm])):
    m = np.flatnonzero(spm & (reg == r)); A_ = m[grp[m] == 'sp_focal']; B_ = m[grp[m] == 'sp_focal2']
    if not len(A_) or not len(B_): continue
    for P_, Q_ in ((A_, B_), (B_, A_)):
        tq = cKDTree(np.c_[gx[Q_], gy[Q_]])
        for i, nb in zip(P_, tq.query_ball_point(np.c_[gx[P_], gy[P_]], 8.0)):
            if not nb: continue
            q = Q_[nb]; o_any[i] = conf[q].max(); s = q[cls[q] == cls[i]]
            if len(s): o_same[i] = conf[s].max()
S_any = conf.astype(np.float64).copy(); S_any[spm] = (conf[spm] + o_any[spm]) / 2
S_same = conf.astype(np.float64).copy(); S_same[spm] = (conf[spm] + o_same[spm]) / 2
stats = dict(n_sp=int(spm.sum()), confirmed_any=int((o_any[spm] > 0).sum()), confirmed_same=int((o_same[spm] > 0).sum()))
print('stats', stats, f'{time.time()-t0:.0f}s', flush=True)
CV = {'A': fast.curves()}
kc = fast.curves(score=S_any, keep=(cls == K0) | (cls == K1)); cv = dict(CV['A']); cv.update({c: kc[c] for c in fast.KUR}); CV['SPF'] = cv
regions = {r for c in CV['A'].values() for r in c.pred_region} | {r for c in CV['A'].values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10)
cnt = {v: honest_counts(c, folds)[0] for v, c in CV.items()}
sa, sb = score_counts(cnt['SPF']), score_counts(cnt['A'])
res = {'A_site': sb[0], 'SPF_site': sa[0], 'd': sa[0] - sb[0], 'boot_family': paired_bootstrap_counts(cnt['SPF'], cnt['A'], site_family, 1000),
       'kur_f1': {c: [sb[1][c]['f1'], sa[1][c]['f1']] for c in fast.KUR}}
json.dump(res, open(OUT / 'res_site_scope.json', 'w'), ensure_ascii=False, indent=1, default=float); print(res)
