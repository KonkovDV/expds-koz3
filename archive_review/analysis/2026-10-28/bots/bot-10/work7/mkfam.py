"""Варианты bot-3 (A2 finds, R3 object_poly, пакет) на семейных файлах bot-10 (work/fam), по правилам eval_pkg.py/eval_r.py bot-3.
python work7/mkfam.py  -> work7/fam_{RB,A2,R3,PKG}/<семья>.jsonl + _counts.json"""
import json, collections, sys
from pathlib import Path
G = Path('/workspace/koz3_shared/bot-3/geo_gap')
SRC = Path('/workspace/koz3_shared/bot-10/work/fam'); OUT = Path(__file__).resolve().parent
FAMS = ['Гнёздово', 'Семибратнее']
def feats(p): return json.load(open(p))['features']
A2 = [f for f in feats(G/'a01/A2.geojson') if f['properties']['class_name'] == 'finds_points']
B = [f for f in feats(G/'a01/B.geojson') if f['properties']['class_name'] == 'finds_points']
R = {n: [f for f in feats(G/f'r/{n}.geojson') if f['properties']['class_name'] == 'object_poly'] for n in ['RB', 'R3']}
def fin_run(r): return 201 <= int(r[:3]) <= 211
def op_run(r, mk): return 201 <= int(r[:3]) <= 217 and not (r.startswith('212') and mk == 'maggr')
VAR = {'FB': (B, None), 'A2': (A2, None), 'RB': (None, 'RB'), 'R3': (None, 'R3'), 'PKG': (A2, 'R3')}
counts = {}
for fam in FAMS:
    base = [json.loads(l) for l in open(SRC/f'{fam}.jsonl')]
    regs = {f['properties']['region_name'] for f in base} | {f['properties']['region_name'] for f in A2 + B + R['R3'] + R['RB'] if f['properties']['region_name'].split('_')[1] == fam}
    for v, (fin, op) in VAR.items():
        out = []; c = collections.Counter()
        for f in base:
            p = f['properties']; r = p['region_name']; mk = p.get('markup_type')
            if fin is not None and p['class_name'] == 'finds_points' and mk == 'maggr' and fin_run(r): c['rm_fin'] += 1; continue
            if op is not None and p['class_name'] == 'object_poly' and op_run(r, mk): c['rm_op'] += 1; continue
            out.append(f)
        if fin is not None:
            add = [f for f in fin if f['properties']['region_name'] in regs and fin_run(f['properties']['region_name'])]
            out += add; c['add_fin'] = len(add)
        if op is not None:
            add = [f for f in R[op] if f['properties']['region_name'] in regs and op_run(f['properties']['region_name'], f['properties']['markup_type'])]
            out += add; c['add_op'] = len(add)
        d = OUT/f'fam_{v}'; d.mkdir(exist_ok=True)
        with open(d/f'{fam}.jsonl', 'w') as fh:
            for f in out: fh.write(json.dumps(f, ensure_ascii=False) + '\n')
        counts[f'{v}|{fam}'] = dict(c); print(v, fam, dict(c), flush=True)
json.dump(counts, open(OUT/'_counts.json', 'w'), ensure_ascii=False, indent=1)
