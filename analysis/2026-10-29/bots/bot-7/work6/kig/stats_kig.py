"""Сводка KIG по образцу bot-10 work6/stats_twins.py + robust_accept-подобный k*.
    python stats_kig.py kig_eval.pkl stats_kig.json"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from honest_bootstrap import boot, fam_index, stats, weighted_class_delta
from expds import metric as M
from expds.naming import site_family
R = pickle.load(open(sys.argv[1], 'rb'))
new = {l.strip() for l in Path('/workspace/koz3_shared/bot-10/repo/notes/regions_las_new.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
HEAVY = ('kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha')
V = sorted({v for v, _ in R})
sc = {(v, l): M.score_counts(R[(v, l)][0])[0] for v, l in R}
fams = fam_index([R[(v, 'k10')][0] for v in V])
B = dict(zip(V, boot([R[(v, 'k10')][0] for v in V], 2000, 0, fams)))
def drop_fams(counts, fs):
    return {c: {r: x for r, x in per.items() if site_family(r) not in fs} for c, per in counts.items()}
out = {}
for v in [x for x in V if x != 'A']:
    a, b = R[(v, 'k10')][0], R[('A', 'k10')][0]
    st = stats(B[v] - B['A']); wc = weighted_class_delta(a, b)
    seeds = [sc[(v, f'seed{s}')] - sc[('A', f'seed{s}')] for s in range(5)]
    # k*: жадно убираем семью с наибольшим вкладом, пока Δ > 0
    fs = sorted({site_family(r) for per in a.values() for r in per}); gone = []; d = sc[(v, 'k10')] - sc[('A', 'k10')]
    while d > 0 and len(gone) < len(fs):
        best = max((f for f in fs if f not in gone), key=lambda f: -(M.score_counts(drop_fams(a, set(gone) | {f}))[0] - M.score_counts(drop_fams(b, set(gone) | {f}))[0]))
        gone.append(best); d = M.score_counts(drop_fams(a, set(gone)))[0] - M.score_counts(drop_fams(b, set(gone)))[0]
    kstar = len(gone) if sc[(v, 'k10')] > sc[('A', 'k10')] else 0
    r = {'d_k10': sc[(v, 'k10')] - sc[('A', 'k10')], 'P': st['p_gt0'], 'ci90': [st['p5'], st['p95']],
         'lofo': sc[(v, 'lofo')] - sc[('A', 'lofo')], 'seeds': seeds, 'seeds_pos': sum(x > 0 for x in seeds),
         'heavy': sum(wc[c]['weighted'] for c in HEAVY if c in wc), 'd_new34': M.score_counts(a, new)[0] - M.score_counts(b, new)[0],
         'per_class_weighted': {c: x['weighted'] for c, x in wc.items() if abs(x['weighted']) > 1e-9},
         'kstar': kstar, 'kstar_removed': gone if kstar else [], 'diag_k10': R[(v, 'k10')][1],
         'per_unit_k10': R[(v, 'k10')][2]}
    r['rule'] = {'P1': r['d_k10'] >= 0.002, 'P2': r['P'] >= 0.85, 'P3': r['lofo'] > 0 and r['seeds_pos'] >= 4,
                 'P4': r['heavy'] >= -0.0005 and r['d_new34'] >= -0.001, 'robust_kstar5_P09': kstar >= 5 and r['P'] >= 0.9}
    out[f'{v}-A'] = r
    print(f'{v}-A: d10={r["d_k10"]:+.6f} P={r["P"]:.3f} CI90=[{r["ci90"][0]:+.5f};{r["ci90"][1]:+.5f}] lofo={r["lofo"]:+.6f} '
          f'seeds={r["seeds_pos"]}/5 [{min(seeds):+.5f}..{max(seeds):+.5f}] heavy={r["heavy"]:+.6f} new34={r["d_new34"]:+.6f} k*={kstar} {gone if kstar else ""}')
    print('   per class', {c: round(x, 6) for c, x in r['per_class_weighted'].items()}, 'rule', r['rule'])
    print('   per unit (in/in_real)', {u: (x.get('in', 0), x.get('in_real', 0)) for u, x in r['per_unit_k10'].items() if x.get('in')})
print('A k10 =', round(sc[('A', 'k10')], 6))
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
