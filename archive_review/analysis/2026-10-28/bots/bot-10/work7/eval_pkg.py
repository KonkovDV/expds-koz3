"""Пакет: finds из A2 (аудит 01, 201–211) + object_poly из R3 (pole_circle 0,05, 201–217; 212/maggr из стека). Против стека."""
import json, sys
sys.argv = ['x']
src = open('eval_r.py').read().split('FP = stack_preds')[0]
exec(src)
from evlib import *
A2 = file_preds('a01/A2.geojson')
RUN = {r for r in REGIONS if r[:3] in [str(n) for n in range(201, 212)]}
FA2 = swap('finds_points', RUN, A2.get('finds_points', []))
P0 = stack_preds(c); FP = stack_preds('finds_points')
out = {}
for scope in ['global', 'site']:
    cB = curves_of({c: P0, 'finds_points': FP}, scope)
    for name, PV in [('A2f+R3', {c: build('R3'), 'finds_points': FA2})]:
        cV = curves_of(PV, scope)
        for unseen in ['drop', 'keep']:
            r = criteria(cV, cB, unseen); key = f'{name}|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], '| opoly', {k: r['per_class_var'][c][k] for k in ('tp','fp','f1')}, '| finds', {k: r['per_class_var']['finds_points'][k] for k in ('tp','fp')}, flush=True)
json.dump(out, open('eval_pkg.json', 'w'), ensure_ascii=False, indent=1, default=float)
