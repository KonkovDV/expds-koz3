"""Шаг 1: stack6 separate (как bot-7 scripts/pkg_fam.py, без twins) только для object_poly/finds_points
по посемейным стекам bot-7/work2/fam_kur (= stack_arch + курганы 0,5 м). Код post — из bot-7/repo (bot7-candidate), только чтение."""
import sys, json
from collections import Counter
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-7/repo')
from expds import post
from shapely.geometry import shape, mapping
FAM = Path('/workspace/koz3_shared/bot-7/work2/fam_kur')
out = open('geo_pkg.jsonl', 'w', encoding='utf-8')
cin, cout = Counter(), Counter()
key = lambda d: f'{d.get("class_name")}|{d.get("threshold_group") or d.get("markup_type")}'
for src in sorted(FAM.glob('*.jsonl')):
    feats = []
    for line in open(src, encoding='utf-8'):
        if '"object_poly"' not in line and '"finds_points"' not in line: continue
        f = json.loads(line); p = dict(f['properties'])
        if p.get('class_name') not in ('object_poly', 'finds_points'): continue
        p['geom'] = shape(f['geometry']); feats.append(p)
    cin.update(map(key, feats))
    feats = post.multipole_object_poly(feats, None)
    feats = post.filter_finds_points_regimes(feats, None)
    cout.update(map(key, feats))
    for d in feats:
        g = d.pop('geom'); d['_fam'] = src.stem
        out.write(json.dumps({'type': 'Feature', 'geometry': mapping(g), 'properties': d}, ensure_ascii=False) + '\n')
out.close()
print(json.dumps({'in': sum(cin.values()), 'out': sum(cout.values()),
                  'changed': {k: [cin[k], cout[k]] for k in sorted(set(cin) | set(cout)) if cin[k] != cout[k]}}, ensure_ascii=False))
