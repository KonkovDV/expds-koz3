"""bot-7: пороги групп *_pix по фолдам k10/LOFO и на всём трейне по кривым bot-10/work7/curves_g_S (только чтение)."""
import sys, json
from pathlib import Path
B = Path('/workspace/koz3_shared/bot-10')
sys.path.insert(0, str(B/'repo')); sys.path.insert(0, str(B/'repo/scripts'))
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
cv = load_curves(B/'work7/curves_g_S')
cv = {k: v for k, v in cv.items() if k in ('finds_points', 'object_poly')}
regions = regions_of(cv)
out = {}
every = set(regions)
res = M.sweep_groups({c: M.curve_where(v, every) for c, v in cv.items()})
out['full'] = {c: {g: gv['threshold'] for g, gv in r['groups'].items() if 'maggr' in g or 'etgr' in g or 'terfl' in g} for c, r in res.items()}
print('full', json.dumps(out['full'], ensure_ascii=False), flush=True)
for ln, k in (('k10', 10), ('lofo', None)):
    folds = M.family_folds(regions, site_family, k)
    rows = []
    for held in folds:
        r = M.sweep_groups({c: M.curve_where(v, every - held) for c, v in cv.items()})
        geo = sorted({site_family(x) for x in held if any(f in x for f in ('Судьбищи', 'Семибратнее', 'Гнёздово'))})
        rows.append({'geo_fams_held': geo, 'taus': {c: {g: gv['threshold'] for g, gv in rr['groups'].items() if g.endswith('_pix')} for c, rr in r.items()}})
    out[ln] = rows
    for x in rows:
        if x['geo_fams_held'] or ln == 'k10':
            print(ln, x['geo_fams_held'], json.dumps(x['taus'], ensure_ascii=False), flush=True)
json.dump(out, open('taus_pix.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('DONE')
