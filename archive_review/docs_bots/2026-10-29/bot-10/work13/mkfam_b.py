"""Задача B: семейные файлы S / C2 / SATf для 3 геосемей по правилу variant_S (work7/mkfam_pix.py), стек — work11/fam0."""
import json, collections
from pathlib import Path
SRC = Path('/workspace/koz3_shared/bot-10/work11/fam0'); OUT = Path(__file__).resolve().parent
PIXF = {'S': '/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson', 'C2': '/workspace/koz3_shared/bot-3/audit/calib/pred_ens2.geojson',
        'SATf': '/workspace/koz3_shared/bot-3/audit/var/SATf/pix_lofo.geojson'}
FAMS = ['Судьбищи', 'Семибратнее', 'Гнёздово']; CL = ['finds_points', 'object_poly']
grp = lambda p: p.get('threshold_group') or p.get('markup_type')
counts = {}
for v, path in PIXF.items():
    PIX = json.load(open(path))['features']
    for fam in FAMS:
        pix = [f for f in PIX if f['properties']['region_name'].split('_')[1] == fam]
        cov = {c: {(f['properties']['region_name'], f['properties']['markup_type']) for f in pix if f['properties']['class_name'] == c} for c in CL}
        out, c = [], collections.Counter()
        for l in open(SRC / f'{fam}.jsonl', encoding='utf-8'):
            f = json.loads(l); p = f['properties']
            if p['class_name'] in CL and (p['region_name'], grp(p)) in cov[p['class_name']]:
                c['rm_' + p['class_name'][:4]] += 1; continue
            out.append(f)
        add = [f for f in pix if f['properties']['class_name'] in CL]; out += add; c['add'] = len(add)
        d = OUT / f'fam_{v}'; d.mkdir(exist_ok=True)
        with open(d / f'{fam}.jsonl', 'w', encoding='utf-8') as fh:
            for f in out: fh.write(json.dumps(f, ensure_ascii=False) + '\n')
        counts[f'{v}|{fam}'] = dict(c); print(v, fam, dict(c), flush=True)
json.dump(counts, open(OUT / '_counts_b.json', 'w'), ensure_ascii=False, indent=1)
