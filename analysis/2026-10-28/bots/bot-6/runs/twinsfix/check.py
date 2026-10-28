"""Сверка починки масштаба с bot-10 diag.log: те же объекты (пороги A k10 из bot-10/work6/taus_A.json, work/fam/*.jsonl,
read-only), настоящая функция post.kurgan_cross_modal_twins ветки bot6-twins-only, dedup ae против li выключен (twins-only)."""
import sys, json, resource
resource.setrlimit(resource.RLIMIT_AS, (2_500_000_000, 2_500_000_000))
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from pathlib import Path
from collections import Counter
from shapely.geometry import shape
from expds import metric as M, post
T = json.load(open('/workspace/koz3_shared/bot-10/work6/taus_A.json'))
tau_of = {r: f['taus'] for f in T['layouts']['k10'] for r in f['held']}
tot = Counter()
for p in sorted(Path('/workspace/koz3_shared/bot-10/work/fam').glob('*.jsonl')):
    feats = []
    for line in open(p, encoding='utf-8'):
        f = json.loads(line); pr = f['properties']
        if pr['class_name'] not in ('kurgany_tselye', 'kurgany_povrezhdennye'): continue
        g = M.threshold_group(pr); t = tau_of.get(pr.get('region_name'), {}).get(g, {}).get(pr['class_name'])
        if t is None or pr['confidence'] < t: continue
        feats.append(dict(pr, geom=shape(f['geometry'])))
    out = post.kurgan_cross_modal_twins(feats, {"dedup_ae_vs_li_m": 0.0})
    c = Counter(f['threshold_group'] for f in out if f.get('threshold_group') in ('ae_twin', 'sp_focal_twin'))
    tot += c; print(p.stem, dict(c), flush=True)
print('TOTAL', dict(tot), 'maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
