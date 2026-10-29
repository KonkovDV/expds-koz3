"""Сводка geo_exact.pkl: Δ по раскладкам, точный бутстреп по 3 геосемьям (все 27 выборок с возвращением), исключение одной семьи."""
import sys, os, json, pickle, itertools
from collections import Counter
sys.path.insert(0, os.path.join(os.environ.get('B10', '/workspace/koz3_shared/bot-10'), 'repo'))
from expds import metric as M
from expds.naming import site_family
D = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'geo_exact.pkl', 'rb')); R, DEN = D['R'], D['DEN']
CL = ('finds_points', 'object_poly'); FAMS = ('Гнёздово', 'Семибратнее', 'Судьбищи')
def part(counts, w=None):
    _, per = M.score_counts(counts, weights=w)
    return sum(M.CLASS_WEIGHTS.get(c, 0) * per[c]['f1'] for c in CL if per[c]['n_gt'] > 0) / DEN
def regw(counts, famw):
    rs = {r for c in counts.values() for r in c}
    return {r: famw.get(site_family(r), 0) for r in rs}
PAIRS = [('S', 'A'), ('s6', 'A'), ('Ss6', 'A'), ('Ss6', 'S'), ('Ss6', 's6')]
out = {}
for pol in ('keep', 'drop'):
    Ls = sorted({l for p, _, l in R if p == pol})
    for a, b in PAIRS:
        d = {L: part(R[(pol, a, L)][0]) - part(R[(pol, b, L)][0]) for L in Ls}
        ca, cb = R[(pol, a, 'k10')][0], R[(pol, b, 'k10')][0]
        boots = []
        for combo in itertools.product(FAMS, repeat=3):
            w = Counter(combo); boots.append(part(ca, regw(ca, w)) - part(cb, regw(cb, w)))
        lofam = {f: part(ca, regw(ca, {g: 1 for g in FAMS if g != f})) - part(cb, regw(cb, {g: 1 for g in FAMS if g != f})) for f in FAMS}
        r = {'d_k10': d['k10'], 'd_lofo': d['lofo'], 'seeds': [d[f'seed{s}'] for s in range(5)],
             'P_boot27': sum(x > 0 for x in boots) / 27, 'boot_min': min(boots), 'boot_max': max(boots), 'without_family': lofam}
        out[f'{pol}|{a}-{b}'] = r
        print(f'{pol} {a}-{b}: k10 {r["d_k10"]:+.6f} lofo {r["d_lofo"]:+.6f} seeds ' + ' '.join(f'{x:+.6f}' for x in r['seeds'])
              + f' | P27={r["P_boot27"]:.3f} [{r["boot_min"]:+.5f};{r["boot_max"]:+.5f}] | без семьи: ' + ', '.join(f'{k} {v:+.5f}' for k, v in lofam.items()))
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'geo_stats.json', 'w'), ensure_ascii=False, indent=1)
