from common import *
import numpy as np
a = pickle.load(open('ws_rr_counts_w5cf.pkl', 'rb')); b = pickle.load(open('ws_rr_counts_w5cnx.pkl', 'rb'))
out = {'vsA': json.load(open('ws_rr_results_w5cf.json'))}
for k in ('RR1', 'RR1s', 'RR2'):
    bt = paired_bootstrap_counts(a[k], b[k], site_family, n=2000)
    d = score_counts(a[k])[0] - score_counts(b[k])[0]
    out[f'{k}_cf_vs_cnx'] = dict(d=d, P=bt['p_gt0'], p5=bt['p5'], p95=bt['p95'])
r = out['vsA']['RR1']; p = out['RR1_cf_vs_cnx']
out['PASS'] = bool(r['PASS'] and r['d_new34'] >= -0.001 and r['d_optical'] >= -0.001 and p['d'] >= 0.003 and p['P'] >= 0.9)
try: out['V5cf'] = json.load(open(W/'work/nested_w5cf.json'))
except Exception as e: out['V5cf'] = str(e)
json.dump(out, open('gate_cf.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ('vsA', 'V5cf')}, default=float)); print('RR1 vsA', {k: r[k] for k in ('d', 'P', 'd_new34', 'd_optical', 'floors_ok', 'PASS')})
