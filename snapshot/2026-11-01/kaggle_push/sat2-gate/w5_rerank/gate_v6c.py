"""V6c gate: honest-10 on W5_DATA table; --vs base_counts.pkl for paired bootstrap.
Optional sat_thr_relax: --scale S --thr-json {class:thr} keeps cand if not in sat groups
or conf >= S * thr[class]."""
from common import *
import fast, argparse, pickle, json
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--save', default='')
ap.add_argument('--vs', default='')
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--out', default='w6c_gate.json')
ap.add_argument('--scale', type=float, default=None, help='sat_thr_relax scale; None = keep all')
ap.add_argument('--thr-json', default='', help='JSON {class_name: default_thr}')
ap.add_argument('--sat-groups', default='sp_det2,sp_or,sp_focal,sp_focal2,sp_arch',
                help='comma threshold_group/markup prefixes subject to scale floor')
a = ap.parse_args()

gt = load(W / 'gt_train_v2.geojson'); new, optical = sets(gt); del gt
keep = None
if a.scale is not None and a.thr_json:
    thr = json.load(open(a.thr_json, encoding='utf-8'))
    T = fast.T
    sat_g = tuple(x for x in a.sat_groups.split(',') if x)
    grp = T['grp']; mk = T.get('mk')
    is_sat = np.zeros(len(T['conf']), dtype=bool)
    for i, g in enumerate(grp):
        gs = str(g or '')
        if any(gs == s or gs.startswith(s) for s in sat_g):
            is_sat[i] = True
        elif mk is not None:
            ms = str(mk[i] or '')
            if any(ms.startswith(s) for s in sat_g):
                is_sat[i] = True
    floor = np.zeros(len(T['conf']))
    for i, cidx in enumerate(T['cls']):
        cname = T['CL'][int(cidx)] if 0 <= int(cidx) < len(T['CL']) else ''
        floor[i] = float(thr.get(cname, 0.0)) * float(a.scale)
    keep = (~is_sat) | (T['conf'] >= floor)
    print(f'keep-mask scale={a.scale}: kept={int(keep.sum())}/{len(keep)} sat={int(is_sat.sum())} sat_kept={int((is_sat & keep).sum())}', flush=True)

s, counts, info, folds = honest(fast.curves(keep=keep), new, optical)
print('honest', s, info, flush=True)
if a.save:
    pickle.dump(dict(s=s, counts=counts), open(W / 'work' / a.save, 'wb'))
if a.vs:
    B = pickle.load(open(a.vs, 'rb')); bc = B['counts']
    bt = paired_bootstrap_counts(counts, bc, site_family, n=a.boot)
    d = s[0] - B['s'][0]; dn = s[1] - B['s'][1]; do = s[2] - B['s'][2]
    _, pc = score_counts(counts); _, pb = score_counts(bc)
    res = dict(
        score=s[0], base=B['s'][0], d=d, d_new34=dn, d_optical=do,
        boot={k: v for k, v in bt.items() if k != 'samples'},
        per_class={c: pc[c]['f1'] - pb[c]['f1'] for c in pc}, info=info,
        scale=a.scale,
        accept=bool(d >= 0.003 and bt['p_gt0'] >= 0.9 and dn >= -0.001 and do >= -0.001),
    )
    # damaged floor (HANDOFF: damaged not worse)
    dam = res['per_class'].get('kurgany_povrezhdennye', 0)
    res['damaged_ok'] = bool(dam >= -1e-9)
    res['pass_full'] = bool(res['accept'] and res['damaged_ok'] and bt['p_gt0'] >= 0.9)
    res['pass_soft'] = bool(d >= 0.003 and bt['p_gt0'] >= 0.8 and res['damaged_ok'] and dn >= -0.001 and do >= -0.001)
    json.dump(res, open(W / 'work' / a.out, 'w'), indent=1, default=float)
    print(json.dumps(res, default=float)[:2000])
