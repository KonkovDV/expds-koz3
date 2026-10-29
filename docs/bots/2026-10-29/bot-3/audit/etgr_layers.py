"""Пункт 4: слои EtGr трейна — коллизии ключей, фильтр глубины, сколько эталонных объектов на каждом слое. → out/etgr_layers.json"""
import sys, json, re, glob, collections
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt/scripts')
import pixgbm as P
from expds import geophys
G = P.G
rows = []; coll = 0
for reg in P.regions('/workspace/data/train-expds-3'):
    files = reg.rasters.get('etgr', [])
    if not files: continue
    cls = set(geophys.select_rasters(files, 'etgr', 'gray')); pix = set(G.pick_rasters(files, None, 'etgr'))
    keys = collections.Counter(G.raster_key(f) for f in files); coll += sum(v - 1 for v in keys.values() if v > 1)
    gt = {}
    for g in glob.glob(f'{reg.path}/*разметка/*EtGr*ObjectPoly.geojson'):
        try: gt[G.label_key(g)] = len(json.load(open(g))['features'])
        except Exception: pass
    for f in files:
        rows.append(dict(region=reg.name, raster=f.name, key=G.raster_key(f), classic=f in cls, pix=f in pix, n_gt=gt.get(G.raster_key(f), None)))
n_gt = sum(r['n_gt'] or 0 for r in rows)
summ = dict(rasters=len(rows), key_collisions=coll, classic_used=sum(r['classic'] for r in rows), pix_used=sum(r['pix'] for r in rows),
            gt_total=n_gt, gt_on_classic_dropped=sum(r['n_gt'] or 0 for r in rows if not r['classic']), gt_on_pix_dropped=sum(r['n_gt'] or 0 for r in rows if not r['pix']),
            rasters_without_gt_match=sum(r['n_gt'] is None for r in rows))
print(json.dumps(summ, ensure_ascii=False))
for r in rows:
    if not r['classic'] or not r['pix'] or r['n_gt'] is None: print(json.dumps(r, ensure_ascii=False))
json.dump(dict(summary=summ, rows=rows), open('/workspace/koz3_shared/bot-3/audit/out/etgr_layers.json', 'w'), ensure_ascii=False, indent=1)
