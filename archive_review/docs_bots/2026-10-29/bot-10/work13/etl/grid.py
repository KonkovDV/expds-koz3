"""Слой классического кандидата EtGr по сетке пикселей: доля вершин на узлах сетки растра (UTM)."""
import rasterio, glob, json, re, collections, numpy as np
from pyproj import Transformer
tr = Transformer.from_crs(3857, 32636, always_xy=True)
R = collections.defaultdict(dict)
for f in sorted(glob.glob('/workspace/data/train-expds-3/21[2-7]_*/*EtGr*/*.tif')):
    reg = f.split('/')[-3]; key = re.sub(r'^\d+_', '', f.split('/')[-1][:-4])
    with rasterio.open(f) as d: R[reg][key] = (~d.transform)
def score(reg, coords, key):
    inv = R[reg][key]; c = 2 * np.array([inv * (x, y) for x, y in coords])
    return float(np.mean(np.all(np.abs(c - np.round(c)) < 0.05, axis=1)))
res = collections.Counter(); best = {}; ex = 0
for l in open('fam_A/Гнёздово.jsonl', encoding='utf-8'):
    f = json.loads(l); p = f['properties']
    if p.get('markup_type') != 'etgr': continue
    ring = f['geometry']['coordinates'][0] if f['geometry']['type'] == 'Polygon' else f['geometry']['coordinates'][0][0]
    xs, ys = tr.transform([c[0] for c in ring], [c[1] for c in ring]); coords = list(zip(xs, ys))[:-1]
    sc = sorted(((score(p['region_name'], coords, k), k) for k in R[p['region_name']]), reverse=True)
    if ex < 3: print(p['region_name'], len(coords), sc[:3]); ex += 1
    top = sc[0][0]; n_top = sum(1 for s, k in sc if s >= top - 1e-9)
    sec = sc[1][0] if len(sc) > 1 else 0; res[('hi' if top > 0.8 else 'mid' if top > 0.4 else 'lo', 'uniq' if top >= 2 * sec + 0.05 else 'amb')] += 1
    best[p['fid']] = [k for s, k in sc if s >= top - 1e-9] if top >= 2 * sec + 0.05 else None
print(sorted(res.items()))
json.dump(best, open('classic_grid_layers.json', 'w'), ensure_ascii=False)
