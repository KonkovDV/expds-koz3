"""V6c gate: honest-10 counts on the current W5_DATA table; --save writes them, --vs <pkl> runs the paired bootstrap
(this - that). Usage: (base dir) gate_v6c.py --save base_counts.pkl ; (cnx dir) gate_v6c.py --vs <base>/work/base_counts.pkl"""
from common import *
import fast, argparse, pickle
ap = argparse.ArgumentParser(); ap.add_argument('--save', default=''); ap.add_argument('--vs', default=''); ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', default='w6c_gate.json'); a = ap.parse_args()
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
s, counts, info, folds = honest(fast.curves(), new, optical); print('honest', s, info, flush=True)
if a.save: pickle.dump(dict(s=s, counts=counts), open(W/'work'/a.save, 'wb'))
if a.vs:
    B = pickle.load(open(a.vs, 'rb')); bc = B['counts']
    bt = paired_bootstrap_counts(counts, bc, site_family, n=a.boot)
    d = s[0] - B['s'][0]; dn = s[1] - B['s'][1]; do = s[2] - B['s'][2]
    _, pc = score_counts(counts); _, pb = score_counts(bc)
    res = dict(score=s[0], base=B['s'][0], d=d, d_new34=dn, d_optical=do, boot={k: v for k, v in bt.items() if k != 'samples'},
               per_class={c: pc[c]['f1'] - pb[c]['f1'] for c in pc}, info=info,
               accept=bool(d >= 0.003 and bt['p_gt0'] >= 0.9 and dn >= -0.001 and do >= -0.001))
    json.dump(res, open(W/'work'/a.out, 'w'), indent=1, default=float); print(json.dumps(res, default=float)[:1500])
