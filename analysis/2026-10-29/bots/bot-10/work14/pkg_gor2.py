# bot-10 work14 (эксплораторно, post-hoc после diag14): где ставить F относительно R.
# PnoD  = пакет, но круги sp_dfine-городищ выключены (τ=inf); PFnoD = PnoD + точки (τ стека от PnoD, τ dense frozen_pick).
# PFj   = пакет + точки, ВСЕ пороги групп (вкл. sp_dfine и gor_dense) совместно sweep_groups на train (данные сами решают про sp_dfine).
# Голова pkg_gor.py до цикла (те же раскладки/машинерия). Выход: pkg_gor2_counts.json.
import os, json, collections, time
from pathlib import Path
_s = Path(__file__).with_name('pkg_gor.py').read_text(encoding='utf-8').split('\nSUBFIX = ')[0]
exec(compile(_s, 'pkg_gor_head', 'exec'))
NOD = {'sp_dfine': np.inf}
C = {}; CH = collections.defaultdict(lambda: collections.defaultdict(list)); t0_ = time.time()
for nm in LAY:
    folds = layouts[nm]; every = set().union(*folds); per = {k: {} for k in ('PnoD', 'PFnoD', 'PFj')}
    for fr, held in zip(PK['layouts'][nm], folds):
        train = every - held; gp_ = gate_of(train, held); rw = rows_of(fr, 'all+cp', gp_)
        cvc = build(True, [], None); tc = taus_of(cvc, train, NOD); held_counts(cvc, held, tc, np.inf, per['PnoD'])
        cv = build(True, rw, gp_); t_d, _ = frozen_pick(cv, train, tc); held_counts(cv, held, tc, t_d, per['PFnoD']); CH[nm]['PFnoD'].append(t_d)
        tj = taus_of(cv, train); held_counts(cv, held, tj, tj.get('gor_dense', np.inf), per['PFj']); CH[nm]['PFj'].append([tj.get('sp_dfine'), tj.get('gor_dense')])
    C[nm] = per; log(nm, f'{time.time()-t0_:.0f}s', {k: gsc(v) for k, v in per.items()})
    json.dump({'C': C, 'chosen': CH}, open(W / 'pkg_gor2_counts.json', 'w'), ensure_ascii=False, default=float)
log('done')
