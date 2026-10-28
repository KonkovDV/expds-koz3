"""Диагностика к gor_scene.py (пост-хок, не для приёмки): AUC по одному признаку, gate без новых строк,
gate с p=const (проверка, что эффект не от масштабирования)."""
import sys
src = open(__file__.replace('gor_scene_diag.py', 'gor_scene.py'), encoding='utf-8').read()
exec(src.split("VARS = [")[0])
fa = {k: auc(X[:, n], y) for n, k in enumerate(FN)}
log('single-feature AUC (all scenes, no model):', {k: round(v, 3) for k, v in sorted(fa.items(), key=lambda t: -abs(t[1] - 0.5))})
D = {'feature_auc': fa}
base_g = {nm: gor_counts(f, None) for nm, f in layouts.items()}
for nm in ['fam10', 'LOFO'] + [f'seed{s}' for s in range(5)]:
    per = gor_counts(layouts[nm], None, True)
    b, v = gsc(base_g[nm]), gsc(per); D.setdefault('gate_only', {})[nm] = dict(var=v, delta=15 * (v['f1'] - b['f1']) / 100)
    log('gate_only', nm, D['gate_only'][nm])
# что меняет gate: какие участки потеряли/получили TP (fam10)
per = gor_counts(folds10, 'csnap', True); bb = base_g['fam10']
diff = {r: (bb.get(r, [0,0,0]), per.get(r, [0,0,0])) for r in set(bb) | set(per) if bb.get(r, [0,0,0])[:2] != per.get(r, [0,0,0])[:2]}
D['csnap_gate_region_diff_fam10'] = {r: {'base': a, 'var': b, 'in_new34': r in new, 'las': r in boxes} for r, (a, b) in sorted(diff.items())}
for r, (a, b) in sorted(diff.items()): log(r, a, '->', b, 'new34' if r in new else '', 'las' if r in boxes else '')
json.dump(D, open(OUT / 'gor_scene_diag.json', 'w'), ensure_ascii=False, indent=1, default=float)
log('done')
