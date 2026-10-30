"""W0 gate G2: decomposition of G1/A_rp by class, against a main-like base B' (kurgans already micro-disk at
centroid-if-inside, as post.micro_circles in configs/default.toml; other classes as in the stack).
Variants: rp-disk for one class at a time on top of B'; NK = rp-disk for all non-kurgan classes on top of B'."""
import sys, json, pickle
sys.path.insert(0, '/workspace/koz3/wt/w5/analysis/2026-10-30/w5_rerank')
from common import *
import fast
NP = pickle.load(open(W/'work/w0_disk_pairs.pkl','rb'))
new, optical = sets(load(W/'gt_train_v2.geojson'))
def PLof(pairs):
    out = {}
    for k,(P,J,S) in pairs.items():
        d = {}
        for p,j,s in zip(P,J,S): d.setdefault(int(p), []).append((int(j), float(s)))
        out[k] = d
    return out
BASE = fast.PL; RP = PLof(NP['rp']); CEN = PLof(NP['cen'])
KUR = {'kurgany_povrezhdennye','kurgany_tselye'}
def mix(rule):
    return {k: (rule(k[0])[k] if rule(k[0]) is not None else BASE[k]) for k in BASE}
def run(PL):
    fast.PL = PL; c = fast.curves(); s, counts, _, _ = honest(c, new, optical); fast.PL = BASE; return s, counts
Bp = lambda pc: CEN if pc in KUR else None
s0, c0 = run(mix(Bp)); res = {'Bprime': s0}; print('Bprime', s0, flush=True)
sb, cb = run(BASE); b = paired_bootstrap_counts(c0, cb, site_family, n=2000, seed=0)
res['Bprime_vs_stack'] = dict(d=s0[0]-sb[0], d_new=s0[1]-sb[1], d_opt=s0[2]-sb[2], p_gt0=b['p_gt0']); print('Bprime_vs_stack', res['Bprime_vs_stack'], flush=True)
V = {'kur_rp': lambda pc: RP if pc in KUR else None}
for c in ['gorodishcha','arkhitektury','object_poly','fortifikatsii']:
    V[c+'_rp'] = (lambda cc: (lambda pc: RP if pc == cc else (CEN if pc in KUR else None)))(c)
V['NK_rp'] = lambda pc: CEN if pc in KUR else (RP if pc != 'finds_points' else None)
for name, rule in V.items():
    s, c = run(mix(rule)); b = paired_bootstrap_counts(c, c0, site_family, n=2000, seed=0)
    res[name] = dict(s=s, d=s[0]-s0[0], d_new=s[1]-s0[1], d_opt=s[2]-s0[2], p_gt0=b['p_gt0'], p5=b['p5'], p95=b['p95'])
    print(name, res[name], flush=True)
json.dump(res, open('g2_decomp.json','w'), indent=1, default=float)
