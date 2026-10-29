"""bot-10 (29.10): сравнение двух наборов порогов *_pix на пуле LOFO-OOF (bot-3/pixgbm/pix_lofo.geojson: каждый участок —
модель без своей семьи), вариант S, официальный compute_metrics.py, точное пересопоставление после порогов (exact_P bot-3).
Набор 9: bot-3/pixgbm/pc/thr_pix_full.json (bot-9, fit_thr.py). Набор 3: bot-3/pixgbm/thresholds_v2.toml.
Бутстреп: все 27 выборок 3 геосемей с возвращением + 2000 случайных (seed 0); без каждой семьи."""
import os, sys, json, itertools, random, tomllib
from collections import Counter
exec(open('/workspace/koz3_shared/bot-3/pixgbm/thr/thr_v2.py').read().split("res = {}")[0])
_b = open('/workspace/koz3_shared/bot-3/pixgbm/thr/thr_v2b.py').read()
exec(_b[_b.index('def exact_P'):_b.index('out = {}')])   # exact_P дословно из thr_v2b.py
os.chdir('/workspace/koz3_shared/bot-10/work9')
J = json.load(open('/workspace/koz3_shared/bot-3/pixgbm/pc/thr_pix_full.json'))
T9 = {(g, c): v for c in CL for g, v in J[f'S|global|{c}'].items()}
TT = tomllib.load(open('/workspace/koz3_shared/bot-3/pixgbm/thresholds_v2.toml', 'rb'))['thresholds_by_markup']
T3 = {(g, c): float(v) for g, d in TT.items() for c, v in d.items()}
print('T9', T9); print('T3', T3, flush=True)
assert set(T9) == set(T3) == set(PG), (T9, T3, PG)
FAMS = sorted({REG_FAM[r] for c in CL for r in CV[c].gt_regions})
res = {'T9': {f'{g}|{c}': v for (g, c), v in T9.items()}, 'T3': {f'{g}|{c}': v for (g, c), v in T3.items()}, 'fams': FAMS}
CS = {}
for n, t in (('stack', None), ('T9', T9), ('T3', T3)):
    cs = exact_P(STACK, {}) if t is None else exact_P(SV, t)
    CS[n] = cs; res[n + '_score'] = total(t or {}, cs); res[n + '_cls'] = {c: f1c(cs[c]) for c in CL}
    print(n, res[n + '_score'], res[n + '_cls'], flush=True)
def sc_w(cs, w):
    full = {}
    for c, per in BASE['counts'].items(): full[c] = per
    full.update(cs)
    rw = {r: w.get(REG_FAM.get(r, site_family(r)), 1 if REG_FAM.get(r, site_family(r)) not in FAMS else 0) for c in full for r in full[c]}
    return M.score_counts(full, weights=rw)[0]
d = lambda w: sc_w(CS['T3'], w) - sc_w(CS['T9'], w)
res['d_T3_T9'] = res['T3_score'] - res['T9_score']
b27 = [d(Counter(cmb)) for cmb in itertools.product(FAMS, repeat=3)]
rnd = random.Random(0); b2k = [d(Counter(rnd.choice(FAMS) for _ in FAMS)) for _ in range(2000)]
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
res['P27_T3gtT9'] = sum(x > 0 for x in b27) / 27; res['P2000_T3gtT9'] = sum(x > 0 for x in b2k) / 2000
res['CI90_2000'] = [q(b2k, 0.05), q(b2k, 0.95)]
res['without_family'] = {f: d({g: 1 for g in FAMS if g != f}) for f in FAMS}
res['per_family_only'] = {f: d({f: 1}) for f in FAMS}
print(json.dumps({k: v for k, v in res.items() if not k.endswith('_cls')}, ensure_ascii=False, default=float), flush=True)
json.dump(res, open('thr_cmp.json', 'w'), ensure_ascii=False, indent=1, default=float)
