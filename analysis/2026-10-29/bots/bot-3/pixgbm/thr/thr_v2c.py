"""Доп. (пост-фактум диагностика): вклад etgr/terfl порогов и S без замены terfl (классика terfl остаётся)."""
exec(open('thr_v2b.py').read().split("out = {}")[0])
out = {}
rec = {**opt, ('maggr_pix', 'finds_points'): 0.93, ('terfl_pix', 'object_poly'): 1.01}
V = {'rec': rec, 'rec_etgr0': {**rec, ('etgr_pix', 'object_poly'): 0.0}, 'rec_terfl0': {**rec, ('terfl_pix', 'object_poly'): 0.0}}
for n, t in V.items():
    cs = exact_P(SV, t); out[n] = [total(t, cs), {c: f1c(cs[c]) for c in CL}]; print(n, out[n], flush=True)
cov_t = {(p[2], 'terfl') for p in PIX['object_poly'] if p[3] == 'terfl_pix'}
SVk = {'finds_points': SV['finds_points'],
       'object_poly': [p for p in SV['object_poly'] if p[3] != 'terfl_pix'] + [p for p in STACK['object_poly'] if (p[2], p[3]) in cov_t]}
cs = exact_P(SVk, rec); out['rec_keep_classic_terfl'] = [total(rec, cs), {c: f1c(cs[c]) for c in CL}]; print('rec_keep_classic_terfl', out['rec_keep_classic_terfl'])
cs = exact_P(STACK, {}); out['stack'] = [total({}, cs), {c: f1c(cs[c]) for c in CL}]; print('stack', out['stack'])
json.dump(out, open('thr_v2c.json', 'w'), ensure_ascii=False, indent=1, default=float)
