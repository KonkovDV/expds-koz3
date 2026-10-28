"""Сводка точной оценки twins: Δ к базе, бутстреп по семьям (2000), LOFO, 5 сидов, тяжёлые классы, new34.
    python work6/stats_twins.py work6/eval_global.pkl work6/stats_global.json"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from honest_bootstrap import boot, fam_index, stats, weighted_class_delta
from expds import metric as M
R = pickle.load(open(sys.argv[1], 'rb'))
new = {l.strip() for l in Path('/workspace/koz3_shared/bot-10/repo/notes/regions_las_new.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
HEAVY = ('kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha')
V = sorted({v for v, _ in R}); Ls = sorted({l for _, l in R})
sc = {(v, l): M.score_counts(R[(v, l)][0])[0] for v, l in R}
out = {'scores': {f'{v}|{l}': s for (v, l), s in sc.items()}, 'twins': {f'{v}|{l}': R[(v, l)][1] for v, l in R}, 'cmp': {}}
fams = fam_index([R[(v, 'k10')][0] for v in V])
B = dict(zip(V, boot([R[(v, 'k10')][0] for v in V], 2000, 0, fams)))
for v, base in [(x, 'A') for x in V if x != 'A' and not x.startswith('kur')] + [('kur', 'A'), ('kurTO', 'kur'), ('kurTOfix', 'kur'), ('kurTO', 'A'), ('kurTOfix', 'A')]:
    if v not in V: continue
    a, b = R[(v, 'k10')][0], R[(base, 'k10')][0]
    st = stats(B[v] - B[base]); wc = weighted_class_delta(a, b)
    _, pa = M.score_counts(a); _, pb = M.score_counts(b)
    seeds = [sc[(v, f'seed{s}')] - sc[(base, f'seed{s}')] for s in range(5)]
    r = {'d_k10': sc[(v, 'k10')] - sc[(base, 'k10')], 'P': st['p_gt0'], 'ci90': [st['p5'], st['p95']], 'ci95': [st['p2_5'], st['p97_5']],
         'lofo': sc[(v, 'lofo')] - sc[(base, 'lofo')], 'seeds': seeds, 'seeds_pos': sum(x > 0 for x in seeds),
         'heavy': sum(wc[c]['weighted'] for c in HEAVY if c in wc), 'd_new34': M.score_counts(a, new)[0] - M.score_counts(b, new)[0],
         'povr_tp_fp': [[pa['kurgany_povrezhdennye']['tp'], pa['kurgany_povrezhdennye']['fp']], [pb['kurgany_povrezhdennye']['tp'], pb['kurgany_povrezhdennye']['fp']]],
         'per_class_weighted': {c: x['weighted'] for c, x in wc.items() if abs(x['weighted']) > 1e-9}}
    r['rule'] = {'P1': r['d_k10'] >= 0.002, 'P2': r['P'] >= 0.85, 'P3': r['lofo'] > 0 and r['seeds_pos'] >= 4,
                 'P4': r['heavy'] >= -0.0005 and r['d_new34'] >= -0.001, 'P5_package_only': r['d_k10'] < 0.006}
    out['cmp'][f'{v}-{base}'] = r
    print(f'{v}-{base}: d10={r["d_k10"]:+.6f} P={r["P"]:.3f} CI90=[{r["ci90"][0]:+.5f};{r["ci90"][1]:+.5f}] lofo={r["lofo"]:+.6f} '
          f'seeds={r["seeds_pos"]}/5 [{min(seeds):+.5f}..{max(seeds):+.5f}] heavy={r["heavy"]:+.6f} new34={r["d_new34"]:+.6f} povr tp/fp={r["povr_tp_fp"][0]} vs {r["povr_tp_fp"][1]}')
print('A k10 =', round(sc[('A', 'k10')], 6), 'twins k10:', {v: R[(v, 'k10')][1] for v in V})
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
