# bot-10 work14: доп. пары по counts из pkg_gor_counts.json + pkg_gor2_counts.json (без пересчёта кривых). Выход pkg_gor2.json.
import os, json
from pathlib import Path
_t = Path(__file__).with_name('pkg_gor.py').read_text(encoding='utf-8')
exec(compile(_t.split('\nSUBFIX = ')[0], 'pkg_gor_head', 'exec'))
C = json.load(open(W / 'pkg_gor_counts.json'))['C']; C2 = json.load(open(W / 'pkg_gor2_counts.json'))
for nm in C: C[nm].update(C2['C'][nm])
C = {nm: {k: {r: tuple(v) for r, v in per.items()} for k, per in d.items()} for nm, d in C.items()}
exec(compile(_t.split('# ---- статистика ----')[1].split('\nPAIRS = ')[0], 'pkg_gor_stats', 'exec'))
SEEDS = [l for l in C if l.startswith('seed')]
R = {'definition': 'bot-10/work14/stats2.py (пары) над pkg_gor_counts.json + pkg_gor2_counts.json (pkg_gor2.py); эксплораторно, post-hoc',
     'counts_fam10': {k: gsc(v) for k, v in C['fam10'].items()}, 'chosen2': C2['chosen'], 'pairs': {}}
for a, b in [('PFnoD', 'P'), ('PFnoD', 'PF'), ('PnoD', 'P'), ('PFnoD', 'PnoD'), ('PFj', 'P'), ('AF', 'P'), ('PFnoD', 'AF')]:
    va, vb = full(C['fam10'][a]), full(C['fam10'][b]); o = {}
    o['fam10_exact'] = score_counts(va)[0] - score_counts(vb)[0]
    o['new34'] = score_counts(va, new)[0] - score_counts(vb, new)[0]; o['optical'] = score_counts(va, optical)[0] - score_counts(vb, optical)[0]
    o['per_heavy'] = perclass(va, vb); o['boot'] = paired_bootstrap_counts(va, vb, site_family, n=1000, seed=0)
    o['LOFO'] = dl('LOFO', a, b); o['seeds'] = {s: dl(s, a, b) for s in SEEDS}
    o['seeds0_4_pos'] = sum(o['seeds'][s] > 0 for s in SEEDS if int(s[4:]) < 5); o['seeds5_9_pos'] = sum(o['seeds'][s] > 0 for s in SEEDS if int(s[4:]) >= 5)
    o['seeds_mean'] = float(np.mean(list(o['seeds'].values())))
    R['pairs'][f'{a}-{b}'] = o; log(f'{a}-{b}', json.dumps({k: o.get(k) for k in ('fam10_exact', 'new34', 'LOFO', 'seeds_mean', 'seeds0_4_pos', 'seeds5_9_pos')}, default=float), o['boot'])
json.dump(R, open(W / 'pkg_gor2.json', 'w'), ensure_ascii=False, indent=1, default=float); log('done')
