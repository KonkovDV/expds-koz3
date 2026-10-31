"""SPF (bot-6): среднее с нулём уверенностей sp_focal/sp_focal2 для курганов, R=8 м. Предрегистрация: bot-6/PREREG_ORSP.md."""
import sys, json, time, pickle
import numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, '/workspace/rerank'); sys.path.insert(0, '/workspace/expds-koz3')
import fast
from common import site_family, family_folds, score_counts, paired_bootstrap_counts, sets, load, W, honest_counts
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
for nm, S in (('SPF', S_any), ('SPF_same', S_same)):
    kc = fast.curves(score=S, keep=(cls == K0) | (cls == K1)); cv = dict(CV['A']); cv.update({c: kc[c] for c in fast.KUR}); CV[nm] = cv
print('curves', f'{time.time()-t0:.0f}s', flush=True)
regions = {r for c in CV['A'].values() for r in c.pred_region} | {r for c in CV['A'].values() for r in c.gt_regions}
def rff(regions, k, seed):
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    nm = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(nm)) % k
    return [set().union(*(fams[n] for n, l in zip(nm, lab) if l == i)) for i in range(k)]
def summary(cnt, ref):
    s = [score_counts(cnt)[0], score_counts(cnt, new)[0], score_counts(cnt, optical)[0]]
    r = [score_counts(ref)[0], score_counts(ref, new)[0], score_counts(ref, optical)[0]]
    pa, pb = score_counts(cnt)[1], score_counts(ref)[1]
    wd = {c: WT[c] * (pa[c]['f1'] - pb[c]['f1']) / 100 for c in CL if abs(pa[c]['f1'] - pb[c]['f1']) > 1e-12}
    tpfp = {c: [pa[c].get('tp'), pa[c].get('fp'), pb[c].get('tp'), pb[c].get('fp')] for c in fast.KUR}
    return dict(score=s, d=s[0] - r[0], d_new=s[1] - r[1], d_opt=s[2] - r[2], wdelta=wd, kur_tp_fp_var_base=tpfp)
def fam_contrib(cnt, ref):
    """Δ без каждой семьи (leave-one-family-out по счётчикам): вклад семьи = Δ − Δ_без_неё."""
    fams = {}
    for by in cnt.values():
        for r in by: fams.setdefault(site_family(r), set()).add(r)
    allr = set().union(*fams.values()); d0 = score_counts(cnt)[0] - score_counts(ref)[0]; out = {}
    for f, rs in fams.items():
        w = {r: 1.0 for r in allr - rs}; out[f] = d0 - (score_counts(cnt, None, w)[0] - score_counts(ref, None, w)[0])
    return sorted(out.items(), key=lambda x: -abs(x[1]))[:8]
res = {'stats': stats, 'prereg_sha256': '936437fc552c8ed1ee068e116421575df8fbb88fed2295fd383e2d542a32f4f5'}
lays = sys.argv[1].split(',') if len(sys.argv) > 1 else ['canon', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4', 'lofo']
for lay in lays:
    folds = family_folds(regions, site_family, 10) if lay == 'canon' else (family_folds(regions, site_family, None) if lay == 'lofo' else rff(regions, 10, int(lay[4:])))
    cnt = {v: honest_counts(cv, folds)[0] for v, cv in CV.items()}; L = {}
    for v in cnt: L[v] = summary(cnt[v], cnt['A'])
    if lay == 'canon':
        for v in ('SPF', 'SPF_same'):
            L[v]['boot'] = paired_bootstrap_counts(cnt[v], cnt['A'], site_family, 1000); L[v]['fam_top'] = fam_contrib(cnt[v], cnt['A'])
        pickle.dump(cnt, open(OUT / 'counts_spf_canon.pkl', 'wb'))
    res[lay] = L
    print(lay, {v: (round(L[v]['score'][0], 6), round(L[v]['d'], 6)) for v in cnt}, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(OUT / 'res_spf.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done', time.time() - t0)
