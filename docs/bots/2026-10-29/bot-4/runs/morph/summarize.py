"""Сводка res_*.json для MORPH.md (только чтение res/)."""
import json, glob, os
R = {}
for L in ['k10', 'lofo', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4']:
    p = f'res/res_{L}.json'
    if os.path.exists(p): R[L] = json.load(open(p))
for L, r in R.items():
    line = [f"{L}: base {r['base']:.6f}"]
    for v in ('C', 'M'):
        x = r[v]; b = x['boot']
        line.append(f"{v} {x['score']:.6f} d={x['d']:+.6f} P={b['p_gt0']:.3f} [{b['p5']:+.5f};{b['p95']:+.5f}] new34={x['new34']:+.6f} opt={x['optical']:+.6f} w={x['weighted']}")
    mc = r.get('M_minus_C'); line.append(f"M-C d={mc['d']:+.6f} P={mc['boot']['p_gt0']:.3f} [{mc['boot']['p5']:+.5f};{mc['boot']['p95']:+.5f}]")
    line.append(f"M top families: {[(f, round(d, 6)) for f, d in r['M']['family_contrib_top6']]}")
    line.append(f"C top families: {[(f, round(d, 6)) for f, d in r['C']['family_contrib_top6']]}")
    print('\n  '.join(line))
