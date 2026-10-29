"""Вариант S PIXGBM (bot-3) на посемейных файлах — та же логика, что bot-10 work7/mkfam_pix.py (variant_S из bot-3 eval_pix.py),
только с путями из аргументов: кандидаты стека того же (участок, метод) класса finds_points/object_poly заменяются pix-кандидатами.
    python mkfam_pix_v2.py --src <fam_A> --pix <pix_lofo.geojson> --out <fam_S>"""
import json, collections, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('--src', required=True); ap.add_argument('--pix', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--fams', default='Судьбищи,Семибратнее,Гнёздово'); a = ap.parse_args()
PIX = json.load(open(a.pix, encoding='utf-8'))['features']
grp = lambda p: p.get('threshold_group') or p.get('markup_type')
classes = ['finds_points', 'object_poly']; counts = {}
d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
for fam in a.fams.split(','):
    pix = [f for f in PIX if f['properties']['region_name'].split('_')[1] == fam]
    cov = {c: {(f['properties']['region_name'], f['properties']['markup_type']) for f in pix if f['properties']['class_name'] == c} for c in classes}
    out, c = [], collections.Counter()
    for l in open(Path(a.src) / f'{fam}.jsonl', encoding='utf-8'):
        f = json.loads(l); p = f['properties']
        if p['class_name'] in classes and (p['region_name'], grp(p)) in cov[p['class_name']]:
            c['rm_' + p['class_name'][:4]] += 1; continue
        out.append(f)
    add = [f for f in pix if f['properties']['class_name'] in classes]
    out += add; c['add'] = len(add)
    with open(d / f'{fam}.jsonl', 'w', encoding='utf-8') as fh:
        for f in out: fh.write(json.dumps(f, ensure_ascii=False) + '\n')
    counts[fam] = dict(c); print('S', fam, dict(c), flush=True)
json.dump(counts, open(d / '_counts_pix.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
