"""Финальный набор порогов (округление к середине плато) — точный перематч."""
exec(open('thr_v2b.py').read().split("out = {}")[0])
fin = {('maggr_pix', 'finds_points'): 0.93, ('maggr_pix', 'object_poly'): 0.69, ('etgr_pix', 'object_poly'): 0.0, ('terfl_pix', 'object_poly'): 1.01}
cs = exact_P(SV, fin); r = [total(fin, cs), {c: f1c(cs[c]) for c in CL}]
xs, f = scan('maggr_pix', 'object_poly', fin)
g = {f'{x:.2f}': float(f[np.searchsorted(xs, x, side='right') - 1]) for x in np.arange(0.60, 0.8001, 0.01)}
print('final', r); print('poly_grid', g)
json.dump(dict(final=r, taus={f'{a}|{b}': v for (a, b), v in fin.items()}, poly_grid=g), open('thr_v2d.json', 'w'), ensure_ascii=False, indent=1)
