import sys, json
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
from expds import metric as M
from expds.naming import site_family
from shapely.geometry import shape
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from pyproj import Transformer
TW = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True); TU = {}
MODE = sys.argv[1] if len(sys.argv) > 1 else 'cos'
T = json.load(open('taus_A.json'))
L = T['layouts']['k10']
tau_of = {}
for f in L:
    for r in f['held']: tau_of[r] = f['taus']
cnt = Counter(); dists = []; dli = []
for p in sorted(Path('/workspace/koz3_shared/bot-10/work/fam').glob('*.jsonl')):
    kept = []
    for line in open(p, encoding='utf-8'):
        f = json.loads(line); pr = f['properties']
        if pr['class_name'] not in ('kurgany_tselye', 'kurgany_povrezhdennye'): continue
        g = M.threshold_group(pr); t = tau_of.get(pr.get('region_name'), {}).get(g, {}).get(pr['class_name'])
        if t is None or pr['confidence'] < t: continue
        c = shape(f['geometry']).centroid
        if MODE == 'utm':
            lon, lat = TW.transform(c.x, c.y); z = int((lon + 180) // 6) + 1
            ux, uy = TU.setdefault(z, Transformer.from_crs('EPSG:4326', f'EPSG:{32600+z}', always_xy=True)).transform(lon, lat)
            kept.append((pr['class_name'], g, pr.get('threshold_group'), pr.get('markup_type'), pr['region_name'], ux, uy, pr['confidence'])); continue
        s = 1.0 if MODE == 'raw' else np.cos(np.arctan(np.sinh(c.y / 6378137.0)))
        kept.append((pr['class_name'], g, pr.get('threshold_group'), pr.get('markup_type'), pr['region_name'], c.x * s, c.y * s, pr['confidence']))
    for k in kept: cnt[(k[0], k[1], k[2], k[3])] += 1
    by = defaultdict(list)
    for k in kept: by[k[4]].append(k)
    for r, ks in by.items():
        aep = [(k[5], k[6]) for k in ks if k[0] == 'kurgany_povrezhdennye' and (k[2] or k[3]) == 'ae']
        lip = [(k[5], k[6]) for k in ks if k[0] == 'kurgany_povrezhdennye' and (k[2] or k[3]) == 'li' and k[7] >= 0.06415]
        ta, tl = (cKDTree(aep) if aep else None), (cKDTree(lip) if lip else None)
        for k in ks:
            if k[0] == 'kurgany_tselye' and (k[2] or k[3]) == 'ae':
                dists.append(ta.query((k[5], k[6]))[0] if ta else 1e9); dli.append(tl.query((k[5], k[6]))[0] if tl else 1e9)
for k, v in sorted(cnt.items(), key=str): print(k, v)
d = np.array(dists); l = np.array(dli)
print('ae tselye kept', len(d), 'ae povr<=0.5', (d <= 0.5).sum(), 'li<=4', (l <= 4).sum(), 'twins', ((d > 0.5) & (l > 4)).sum())
print('ae dist quantiles', np.percentile(d[d < 1e8], [5, 25, 50, 75]) if (d < 1e8).any() else None)
