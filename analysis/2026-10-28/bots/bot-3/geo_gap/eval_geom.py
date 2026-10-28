import json
from shapely.ops import polylabel
from evlib import *
c = 'object_poly'
P0 = stack_preds(c)
def pole(g):
    try: return polylabel(g, tolerance=0.05)
    except Exception: return g.representative_point()
def disc(pt, r): return pt.buffer(r, quad_segs=8)
def V(name):
    if name == 'G1': return P0 + [(disc(pole(p[0]), 1.0), p[1], p[2], p[3] + '_pole', p[4]) for p in P0 if p[0].area > 12.6]
    if name == 'G2': return P0 + [(disc(pole(p[0]), 0.5), p[1], p[2], p[3] + '_pole', p[4]) for p in P0 if p[0].area > 3.14]
    if name == 'G3': return [((disc(pole(p[0]), 1.0),) + p[1:]) if p[0].area > 50 else p for p in P0]
FP = stack_preds('finds_points')
out = {}
for scope in ['global', 'site']:
    cB = curves_of({c: P0, 'finds_points': FP}, scope)
    for name in ['G1', 'G2', 'G3']:
        cV = curves_of({c: V(name), 'finds_points': FP}, scope)
        for unseen in ['drop', 'keep']:
            r = criteria(cV, cB, unseen); key = f'{name}|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], '| opoly', {k: r['per_class_var'][c][k] for k in ('tp','fp','f1')}, 'vs', {k: r['per_class_base'][c][k] for k in ('tp','fp','f1')}, flush=True)
json.dump(out, open('eval_geom.json', 'w'), ensure_ascii=False, indent=1, default=float)
