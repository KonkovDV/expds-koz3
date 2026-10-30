import sys, json, collections, numpy as np, torch
sys.argv_ = sys.argv; import ae_rerun as A
from pyproj import Transformer
from shapely.geometry import shape
from transformers import DFineForObjectDetection
from pathlib import Path
oof = json.load(open('/workspace/koz3/out/w1-dfine-ae-final/oof_dfine_ae_noflip.geojson'))['features']
OUT = Path('/workspace/koz3/w1/v8/out_screen'); OUT.mkdir(exist_ok=True); models = {}
for r in sys.argv[1].split(','):
    if (OUT / f'{r}.json').exists(): continue
    inv = A.inv_of(r); pts = collections.defaultdict(list)
    for f in oof:
        p = f['properties']
        if p['region_name'] == r and p['confidence'] >= 0.4: c = shape(f['geometry']).centroid; pts[p['tif']].append((c.x, c.y))
    pick = {}
    for key, size in inv['tifs']:
        sc = A.Scene(inv, key, size); tr = Transformer.from_crs('EPSG:3857', sc.crs, always_xy=True); inv_t = ~sc.T
        P = [inv_t * tr.transform(x, y) for x, y in pts[key.rsplit('/', 1)[-1]]]
        wins = A._wins(sc.W, sc.H, 640, 160)
        cnt = sorted(((sum(1 for px, py in P if x <= px < x + 640 and y <= py < y + 640), (x, y)) for x, y in wins), reverse=True)[:3]
        pick[key] = {w for _, w in cnt}; print(r, key.rsplit('/',1)[-1], 'top OOF counts', [c for c, _ in cnt], flush=True); sc.close()
    k = A.FOLDMAP[r]
    if k not in models:
        models.clear(); models[k] = DFineForObjectDetection.from_pretrained(f'/workspace/koz3/out/w1-dfine-ae-final/dfine_f{k}', device_map=None).eval()
    out, ws = A.infer_region(models[k], inv, torch.device('cpu'), win_filter=lambda key, w: w in pick[key])
    json.dump({'region': r, 'fold': k, 'wstats': ws}, open(OUT / f'{r}.json', 'w'), ensure_ascii=False)
    print(r, 'max n04', max(w['n04'] for w in ws), 'max s100', round(max(w['s100'] for w in ws), 3), flush=True)
