"""V6c gate: honest-10 counts on the current W5_DATA table; --save writes them, --vs <pkl> runs the paired bootstrap
(this - that). Usage: (base dir) gate_v6c.py --save base_counts.pkl ; (cnx dir) gate_v6c.py --vs <base>/work/base_counts.pkl"""
from common import *
import fast, argparse, pickle
ap = argparse.ArgumentParser(); ap.add_argument('--save', default=''); ap.add_argument('--vs', default=''); ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', default='w6c_gate.json')
# W1 sat2 (реконструкция bot-W1-aeor 01.11): keep-маска sat_thr_relax — кандидаты sp_det2 курганов с conf < scale×thr[класс] не выдаются
ap.add_argument('--scale', type=float, default=None); ap.add_argument('--thr-json', default=''); a = ap.parse_args()
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
keep = None
if a.scale is not None:
    import numpy as np
    thr = json.load(open(a.thr_json, encoding='utf-8')); T = fast.T
    cn = np.array(T['CL'], dtype=object)[np.maximum(T['cls'], 0)]
    tv = np.array([float(thr.get(c, 0.0)) if isinstance(thr.get(c, 0.0), (int, float)) else 0.0 for c in cn])
    sat = (T['grp'] == 'sp_det2') & np.isin(cn, fast.KUR) & (T['cls'] >= 0)
    keep = ~(sat & (T['conf'] < a.scale * tv)); print('keep-mask scale', a.scale, 'sat', int(sat.sum()), 'dropped', int((~keep).sum()), flush=True)
s, counts, info, folds = honest(fast.curves(keep=keep), new, optical); print('honest', s, info, flush=True)
if a.save: pickle.dump(dict(s=s, counts=counts), open(W/'work'/a.save, 'wb'))
if a.vs:
    B = pickle.load(open(a.vs, 'rb')); bc = B['counts']
    bt = paired_bootstrap_counts(counts, bc, site_family, n=a.boot)
    d = s[0] - B['s'][0]; dn = s[1] - B['s'][1]; do = s[2] - B['s'][2]
    _, pc = score_counts(counts); _, pb = score_counts(bc)
    res = dict(score=s[0], base=B['s'][0], d=d, d_new34=dn, d_optical=do, boot={k: v for k, v in bt.items() if k != 'samples'},
               per_class={c: pc[c]['f1'] - pb[c]['f1'] for c in pc}, info=info,
               accept=bool(d >= 0.003 and bt['p_gt0'] >= 0.9 and dn >= -0.001 and do >= -0.001))
    dmg = res['per_class'].get('kurgany_povrezhdennye', 0.0) >= 0
    res.update(scale=a.scale, damaged_not_worse=bool(dmg), pass_full=bool(res['accept'] and dmg),
               pass_soft=bool(d >= 0.003 and bt['p_gt0'] >= 0.8 and dn >= -0.001 and do >= -0.001 and dmg))
    json.dump(res, open(W/'work'/a.out, 'w'), indent=1, default=float); print(json.dumps(res, default=float)[:1500])
