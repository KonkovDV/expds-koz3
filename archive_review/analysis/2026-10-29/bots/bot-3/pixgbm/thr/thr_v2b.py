"""Доп. к thr_v2.py: база стека, вложенный LOFO-балл с порогами *_pix, варианты наборов порогов (точный перематч)."""
exec(open('thr_v2.py').read().split("res = {}")[0])
R = json.load(open('thr_v2.json'))
pk = lambda d: {tuple(k.split('|')): v for k, v in d.items()}
opt = pk(R['taus_opt'])
def exact_P(preds, t):
    P = {c: [p for p in preds[c] if p[1] >= t.get((p[3], c), FIX.get((p[3], c), 0.0))] for c in CL}
    cv = curves_of(P, 'global'); cs = {}
    for c in CL:
        per = {r: [0, 0, int(n)] for r, n in cv[c].gt_regions.items()}
        for r, h in zip(cv[c].pred_region, cv[c].hit): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        cs[c] = per
    return cs
out = {}
cs = exact_P(STACK, {}); out['stack'] = [total({}, cs), {c: f1c(cs[c]) for c in CL}]
V = {'default_0': {k: 0.0 for k in PG},
     'opt': opt,
     'maggr_only': {**{k: 0.0 for k in PG}, **{k: v for k, v in opt.items() if k[0] == 'maggr_pix'}},
     'opt_finds093': {**opt, ('maggr_pix', 'finds_points'): 0.93},
     'maggr_only_finds093': {**{k: 0.0 for k in PG}, ('maggr_pix', 'finds_points'): 0.93, ('maggr_pix', 'object_poly'): opt[('maggr_pix', 'object_poly')]},
     'opt_terfl070': {**opt, ('terfl_pix', 'object_poly'): 0.70},
     'rec': {**opt, ('maggr_pix', 'finds_points'): 0.93, ('terfl_pix', 'object_poly'): 0.70}}
for name, t in V.items():
    cs = exact_P(SV, t); out[name] = [total(t, cs), {c: f1c(cs[c]) for c in CL}, {f'{g}|{c}': v for (g, c), v in t.items()}]
    print(name, out[name][0], out[name][1], flush=True)
# вложенный LOFO по семьям: порог семьи F — оптимум на остальных (как honest_counts, unseen=keep → 0)
lofo = {}
for fam, tt in R['lofo_train_taus'].items():
    t = pk(tt); regs = {r for r in REGIONS if REG_FAM[r] == fam}
    lofo[fam] = kept_counts(t, regs)
cs = {c: {} for c in CL}
for fam in lofo:
    for c in CL: cs[c].update(lofo[fam][c])
for c in CL:   # участки вне трёх геосемей — при пороге opt (там pix нет)
    for r, v in kept_counts(opt)[c].items(): cs[c].setdefault(r, v)
out['lofo_nested_fast'] = [total({}, cs), {c: f1c(cs[c]) for c in CL}]
print('lofo_nested_fast', out['lofo_nested_fast'], flush=True)
# мелкая сетка finds
xs, f = scan('maggr_pix', 'finds_points', opt)
out['finds_grid'] = {f'{x:.3f}': float(f[np.searchsorted(xs, x, side='right') - 1]) for x in np.arange(0.80, 0.9901, 0.01)}
xs, f = scan('terfl_pix', 'object_poly', opt)
out['terfl_grid'] = {f'{x:.2f}': float(f[np.searchsorted(xs, x, side='right') - 1]) for x in np.arange(0.0, 1.0001, 0.05)}
xs, f = scan('etgr_pix', 'object_poly', opt)
out['etgr_grid'] = {f'{x:.2f}': float(f[np.searchsorted(xs, x, side='right') - 1]) for x in np.arange(0.0, 1.0001, 0.05)}
# сколько pix-кандидатов на группу проходит
out['n_kept'] = {name: {f'{g}|{c}': int(sum(1 for p in SV[c] if p[3] == g and p[1] >= t[(g, c)])) for (g, c) in PG} for name, t in V.items()}
json.dump(out, open('thr_v2b.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: out[k] for k in ('finds_grid', 'terfl_grid', 'etgr_grid', 'n_kept')}, ensure_ascii=False, default=float))
