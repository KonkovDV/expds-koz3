"""H1–H3 (якоря) на кандидатах стека и пакет с A2 (audit01). П1–П5 против стека."""
import sys, json
from evlib import *
import anchors
c = 'object_poly'
P0 = stack_preds(c); FP = stack_preds('finds_points')
RUN = {r for r in REGIONS if r[:3] in [str(n) for n in range(201, 212)]}
A2 = file_preds('a01/A2.geojson')
A2 = {k: swap(k, RUN, A2.get(k, [])) for k in GEO}
out = {}
for scope in ['global', 'site']:
    cB = curves_of({c: P0, 'finds_points': FP}, scope)
    todo = [(m, {c: anchors.apply(P0, m), 'finds_points': FP}) for m in ['H1', 'H2', 'H3']]
    todo.append(('A2', A2)); todo.append(('A2+H1', {c: anchors.apply(A2[c], 'H1'), 'finds_points': A2['finds_points']}))
    for name, PV in todo:
        cV = curves_of(PV, scope)
        for unseen in ['drop', 'keep']:
            r = criteria(cV, cB, unseen); key = f'{name}|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], '| opoly', {k: r['per_class_var'][c][k] for k in ('tp','fp','f1')}, 'vs', {k: r['per_class_base'][c][k] for k in ('tp','fp','f1')}, flush=True)
json.dump(out, open('eval_anchor.json', 'w'), ensure_ascii=False, indent=1, default=float)
