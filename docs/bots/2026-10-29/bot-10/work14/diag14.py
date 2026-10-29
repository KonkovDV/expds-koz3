# bot-10 work14: fam10 — TP/FP по источникам (stack / sp_dfine / gor_dense) для A, AF, P, PF; голова pkg_gor.py до цикла.
import os, json, collections
from pathlib import Path
_s = Path(__file__).with_name('pkg_gor.py').read_text(encoding='utf-8').split('\nSUBFIX = ')[0]
exec(compile(_s, 'pkg_gor_head', 'exec'))
folds = layouts['fam10']; every = set().union(*folds); out = collections.defaultdict(collections.Counter); NEWc = collections.defaultdict(collections.Counter)
def acc(cv, held, taus, t_d, key):
    hc = curve_where(cv, held); g = np.asarray(hc.pred_group, dtype=object)
    tau = np.array([t_d if x == 'gor_dense' else taus.get(x, np.inf) for x in g]); kept = hc.conf >= tau
    for gg, h, r in zip(g[kept], hc.hit[kept], np.asarray(hc.pred_region, dtype=object)[kept]):
        src = 'dense' if gg == 'gor_dense' else ('dfine' if gg.startswith('sp_dfine') else 'stack'); out[key][f'{src}_{"tp" if h else "fp"}'] += 1
        if r in NEW: NEWc[key][f'{src}_{"tp" if h else "fp"}'] += 1
for fr, held in zip(PK['layouts']['fam10'], folds):
    train = every - held; gp_ = gate_of(train, held); rw = rows_of(fr, 'all+cp', gp_)
    cv0 = build(False, [], None); acc(cv0, held, taus_of(cv0, train), np.inf, 'A')
    tc = taus_of(build(False, [], gp_), train); cv = build(False, rw, gp_); acc(cv, held, tc, frozen_pick(cv, train, tc)[0], 'AF')
    cvp = build(True, [], None); tp_ = taus_of(cvp, train); acc(cvp, held, tp_, np.inf, 'P')
    cv = build(True, rw, gp_); acc(cv, held, tp_, frozen_pick(cv, train, tp_)[0], 'PF')
json.dump({'all': out, 'new34': NEWc}, open(W / 'diag14_fam10.json', 'w'), ensure_ascii=False, indent=1)
for k in out: log(k, dict(sorted(out[k].items())), 'new34', dict(sorted(NEWc[k].items())))
