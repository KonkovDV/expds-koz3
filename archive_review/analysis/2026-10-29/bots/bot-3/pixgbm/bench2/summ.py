"""Сводка bench2 (bot-3): суммы по 7 участкам, ускорения, совпадение кандидатов, пики RSS."""
import json, pickle, collections
from shapely import wkb
R = "201 204 208 212 216 217 200".split()
rows = [json.loads(l) for l in open('bench2.jsonl')]
S = collections.OrderedDict()
for r in rows:
    k = (r['backend'], r['threads'], r['procs']); s = S.setdefault(k, collections.Counter())
    for f in ('wall', 'cpu', 'pred', 'feat_wait', 'n_cand'): s[f] += r[f]
    s['max_rss_mb'] = max(s['max_rss_mb'], r['maxrss_mb'] + r['maxrss_children_mb'])
out = {}
for k, s in S.items():
    b = S[('lightgbm', k[1], 1)]
    out['%s_nt%d_p%d' % k] = d = {f: round(v, 2) for f, v in s.items()}
    d['x_wall'] = round(b['wall'] / s['wall'], 2); d['x_pred'] = round(b['pred'] / s['pred'], 2); d['x_cpu'] = round(b['cpu'] / s['cpu'], 2)
    print(k, d)
def key(p):
    return sorted(pickle.load(open(p, "rb")))
same = {}
for r in R:
    ref = key(f'cand/{r}_lgb_1.pkl')
    for t in ('ll_1', 'lgb_2', 'll_2', 'll_2p2', 'tl_2'):
        same[f'{r}_{t}'] = key(f'cand/{r}_{t}.pkl') == ref
print('cand identical to lgb_1:', all(same.values()), [k for k, v in same.items() if not v])
pr = [json.loads(l) for l in open('preset.jsonl')]
P = {}
for r in pr:
    s = P.setdefault(r['threads'], collections.Counter())
    for f in ('wall', 'cpu', 'pred', 'n_cand'): s[f] += r[f]
print('preset etgr,terfl lleaves:', {k: {f: round(v, 2) for f, v in s.items()} for k, s in P.items()})
json.dump(dict(totals=out, cand_same=same, preset={k: dict(s) for k, s in P.items()}), open('summ.json', 'w'), indent=1, ensure_ascii=False)
