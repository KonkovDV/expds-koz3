"""W0 gate G1: replace every polygon pred by a 5 cm disk at an anchor (rp | centroid-in-P | polylabel),
pairs recomputed in m1_anchor.py. Honest-10 (family folds) + paired family bootstrap vs base. Prereg: 3 variants
(A=rp all, B=polylabel for non-kurgan polygon classes + rp for kurgans, C=centroid all); accept d>=0.003, p_gt0>=0.9 (Bonferroni x3 noted)."""
import os, sys, json, pickle, numpy as np
sys.path.insert(0, '/workspace/koz3/wt/w5/analysis/2026-10-30/w5_rerank')
from common import *
import fast
NP = pickle.load(open(W/'work/w0_disk_pairs.pkl','rb'))
gt = {'features': []}
new, optical = sets(load(W/'gt_train_v2.geojson'))
def PLof(pairs):
    out = {}
    for k,(P,J,S) in pairs.items():
        d = {}
        for p,j,s in zip(P,J,S): d.setdefault(int(p), []).append((int(j), float(s)))
        out[k] = d
    return out
BASE_PL = fast.PL
def run(PL):
    fast.PL = PL; c = fast.curves(); s, counts, info, folds = honest(c, new, optical); fast.PL = BASE_PL; return s, counts
s0, c0 = run(BASE_PL); print('base', s0, flush=True)
KUR = {'kurgany_povrezhdennye','kurgany_tselye'}
var = {'A_rp': PLof(NP['rp']), 'C_cen': PLof(NP['cen']),
       'B_pl_nonkur': PLof({k: (NP['rp'][k] if k[0] in KUR else NP['pl'][k]) for k in NP['rp']})}
res = {'base': s0}
for name, PL in var.items():
    s, c = run(PL)
    b = paired_bootstrap_counts(c, c0, site_family, n=2000, seed=0)
    bn = paired_bootstrap_counts(c, c0, site_family, n=2000, seed=0, regions=new)
    res[name] = dict(s=s, d=s[0]-s0[0], d_new=s[1]-s0[1], d_opt=s[2]-s0[2], boot=b, boot_new=bn)
    print(name, res[name]['d'], res[name]['d_new'], res[name]['d_opt'], {k: b[k] for k in b if not isinstance(b[k], (list, dict))}, flush=True)
json.dump(res, open('g1_disk_gate.json','w'), indent=1, default=float)
