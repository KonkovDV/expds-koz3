"""Варианты pixgbm bot-3 (S, Sf, Sf+R3) на семейных файлах bot-10, по правилу variant_S из bot-3/pixgbm/eval_pix.py:
кандидаты стека того же (участок, метод) класса заменяются pix-кандидатами (pix_lofo.geojson, LOFO-модели)."""
import json, collections
from pathlib import Path
SRC = Path('/workspace/koz3_shared/bot-10/work/fam'); OUT = Path(__file__).resolve().parent
PIX = json.load(open('/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson'))['features']
FAMS = ['Судьбищи', 'Семибратнее', 'Гнёздово']
grp = lambda p: p.get('threshold_group') or p.get('markup_type')
counts = {}
for fam in FAMS:
    pix = [f for f in PIX if f['properties']['region_name'].split('_')[1] == fam]
    cov = {c: {(f['properties']['region_name'], f['properties']['markup_type']) for f in pix if f['properties']['class_name'] == c} for c in ['finds_points', 'object_poly']}
    for v, classes, opsrc in [('Sf', ['finds_points'], None), ('S', ['finds_points', 'object_poly'], None), ('SfR3', ['finds_points'], 'R3')]:
        src = OUT/f'fam_{opsrc}'/f'{fam}.jsonl' if opsrc and (OUT/f'fam_{opsrc}'/f'{fam}.jsonl').exists() else SRC/f'{fam}.jsonl'
        out, c = [], collections.Counter()
        for l in open(src):
            f = json.loads(l); p = f['properties']
            if p['class_name'] in classes and (p['region_name'], grp(p)) in cov[p['class_name']]:
                c['rm_' + p['class_name'][:4]] += 1; continue
            out.append(f)
        add = [f for f in pix if f['properties']['class_name'] in classes]
        out += add; c['add'] = len(add); c['src_' + src.parent.name] = 1
        d = OUT/f'fam_{v}'; d.mkdir(exist_ok=True)
        with open(d/f'{fam}.jsonl', 'w') as fh:
            for f in out: fh.write(json.dumps(f, ensure_ascii=False) + '\n')
        counts[f'{v}|{fam}'] = dict(c); print(v, fam, dict(c), flush=True)
json.dump(counts, open(OUT/'_counts_pix.json', 'w'), ensure_ascii=False, indent=1)
