"""W2: детализация honest-10 по счётчикам gate_v6c (--save): Δ, бутстрап по семьям, полы по классам (n_gt>=5),
срезы new34 / optical / lidar59 (+ per-class в срезе при n_gt>=5), per-site Δ на LAS-участках.
usage (из w5_rerank, W5_DATA=любой): gate_detail.py <base_counts.pkl> <var_counts.pkl> <out.json>"""
from common import *
import pickle, numpy as np
a, b, out = sys.argv[1:4]
A = pickle.load(open(a, 'rb')); B = pickle.load(open(b, 'rb')); cA, cB = A['counts'], B['counts']
gt = load(W / 'gt_train_v2.geojson'); new, optical = sets(gt); del gt
las59 = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))
FLOOR = {'kurgany_povrezhdennye': -0.002, 'kurgany_tselye': -0.002, 'gorodishcha': -0.002}
def sl(regs=None):
    sa, pa = score_counts(cA, regs); sb, pb = score_counts(cB, regs)
    d = {c: dict(dF1=pb[c]['f1'] - pa[c]['f1'], n_gt=pa[c]['n_gt'], tpA=pa[c]['tp'], fpA=pa[c]['fp'], tpB=pb[c]['tp'], fpB=pb[c]['fp']) for c in pa}
    viol = {c: v['dF1'] for c, v in d.items() if v['n_gt'] >= 5 and v['dF1'] < FLOOR.get(c, -0.005)}
    return dict(A=sa, B=sb, d=sb - sa, per_class=d, floor_violations=viol, excluded_lt5=[c for c, v in d.items() if v['n_gt'] < 5])
bt = paired_bootstrap_counts(cB, cA, site_family, n=2000)
res = dict(all=sl(), new34=sl(new), optical=sl(optical), lidar59=sl(las59), boot={k: v for k, v in bt.items() if k != 'samples'})
site = {}
for r in sorted(las59):
    sa = score_counts(cA, {r})[0]; sb = score_counts(cB, {r})[0]
    if abs(sb - sa) > 1e-12: site[r] = sb - sa
res['las_site_d'] = dict(sorted(site.items(), key=lambda x: x[1]))
res['las_sites_drop'] = {r: v for r, v in site.items() if v < 0}
res['gate'] = dict(d_ok=res['all']['d'] >= 0.003, p_ok=bt['p_gt0'] >= 0.9, floors_ok=not res['all']['floor_violations'],
                   new34_ok=res['new34']['d'] >= -0.001, optical_ok=res['optical']['d'] >= -0.001)
res['gate']['PASS'] = all(res['gate'].values())
json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: res[k] for k in ('boot', 'gate')}, default=float)); print('d', res['all']['d'], 'new34', res['new34']['d'], 'opt', res['optical']['d'], 'viol', res['all']['floor_violations'])
