"""Растровые варианты object_poly (bot3-geophys) — честно, П1–П5. 201–217; 200 и 212/maggr — из стека."""
import sys, json, collections
from evlib import *
c = 'object_poly'
REGS = {l for l in os.listdir('/workspace/data/train-expds-3') if l[:3].isdigit() and 201 <= int(l[:3]) <= 217}
import re
def bad_regions(name):
    """участки, где в логе варианта были ошибки файлов (ошибки идут перед строкой 'участок: N'), кроме известной 212/MagGr цвет_2"""
    bad, err = set(), False
    for l in open(f'r/{name}.log'):
        if 'ошибка' in l and not ('Усадьба_MagGr_цвет_2' in l): err = True
        m = re.match(r'^(2\d\d_\S+): \d+$', l.strip())
        if m:
            if err: bad.add(m.group(1))
            err = False
    return bad
EXCL = set()
def keep_stack(p): return p[2] not in REGS or p[2] in EXCL or (p[2].startswith('212') and p[3] == 'maggr')
import anchors
def build(name):
    if name.endswith('H1'): return anchors.apply(build(name[:-2]), 'H1')
    fp = file_preds(f'r/{name}.geojson').get(c, [])
    new = [p for p in fp if not keep_stack(p)]
    return [p for p in stack_preds(c) if keep_stack(p)] + new
FP = stack_preds('finds_points')
PB = build('RB')
# проверка: RB против стека по (участок, группа)
st = collections.Counter((p[2][:3], p[3]) for p in stack_preds(c) if not keep_stack(p))
rb = collections.Counter((p[2][:3], p[3]) for p in PB if not keep_stack(p))
print('RB==stack по числу:', st == rb, {k: (st[k], rb[k]) for k in set(st) | set(rb) if st[k] != rb[k]})
out = {}
for scope in ['global', 'site']:
    cS = curves_of({c: stack_preds(c), 'finds_points': FP}, scope)
    cB = curves_of({c: PB, 'finds_points': FP}, scope)
    for name in ['RB'] + [n for n in sys.argv[1:] if os.path.exists(f'r/{n.removesuffix("H1")}.geojson')]:
        EXCL.clear(); EXCL.update(bad_regions(name.removesuffix('H1')) | bad_regions('RB'))
        if EXCL: print(name, 'участки из стека (ошибки файлов):', sorted(EXCL))
        cB = curves_of({c: build('RB'), 'finds_points': FP}, scope)
        cV = cB if name == 'RB' else curves_of({c: build(name), 'finds_points': FP}, scope)
        for unseen in ['drop', 'keep']:
            r = criteria(cV, cS if name == 'RB' else cB, unseen); key = f'{name}|{scope}|{unseen}'; out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], '| opoly', {k: r['per_class_var'][c][k] for k in ('tp','fp','f1')}, 'vs', {k: r['per_class_base'][c][k] for k in ('tp','fp','f1')}, flush=True)
json.dump(out, open('r/eval_r.json', 'w'), ensure_ascii=False, indent=1, default=float)
