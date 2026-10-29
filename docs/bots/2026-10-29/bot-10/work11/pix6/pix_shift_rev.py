"""обратный случай: тест ведёт себя как OOF (сдвига нет), а τ подняли/сдвинули квантильно (τ' = F_full^-1(F_oof(τ))) — цена ненужного сдвига.
Также «сдвиг на X» только для maggr_pix finds_points. Импортирует определения pix_shift.py (без записи в bot-3). Пишет pix_shift_rev.json."""
import os, sys, json
os.environ['NO_METRIC'] = '1'
code = open('pix_shift.py').read()
head, tail = code.split("if os.environ.get('NO_METRIC'): sys.exit(0)")
exec(head.replace("open(f'{OUTD}/pix_shift_dist.json', 'w')", "open(os.devnull, 'w')"))
mtail = tail.split('out = {}')[0]; exec(mtail)
_orig = g_inv
def g_inv(key, kind):
    if kind == 'q': return _orig(key, 'q')
    lo2, fu2 = MAPS[key]
    if kind == 'rev': return lambda t: qmatch(lo2, fu2, t)
    if kind.startswith('fp+'):  # сдвиг только finds на X
        X = float(kind[3:]); return (lambda t: t + X) if key == ('maggr_pix', 'finds_points') else (lambda t: t)
out = {}
for scope in ['global']:
    cv = curves_of(VARS['S'], scope)
    base, _ = honest_counts_shift(cv, FOLDS, 'keep', None); sb = scores(base)
    for kind in ['rev', 'fp+0.02', 'fp+0.04']:
        var, used = honest_counts_shift(cv, FOLDS, 'keep', kind); sv = scores(var)
        pb = M.paired_bootstrap_counts(var, base, site_family, n=1000)
        pa, pbb = M.score_counts(var)[1], M.score_counts(base)[1]
        out[f'{scope}|keep|{kind}'] = {'base': sb, 'var': sv, 'd': sv[0] - sb[0], 'd_new34': sv[1] - sb[1], 'boot': pb,
            'per_class': {c: [(pbb[c]['tp'], pbb[c]['fp']), (pa[c]['tp'], pa[c]['fp'])] for c in CL}}
        print(scope, kind, [round(x, 6) for x in sb], [round(x, 6) for x in sv], 'Δ %+.6f P(>0) %.3f p5 %+.5f p95 %+.5f' % (sv[0] - sb[0], pb['p_gt0'], pb['p5'], pb['p95']), out[f'{scope}|keep|{kind}']['per_class'], flush=True)
json.dump(out, open('pix_shift_rev.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done')
