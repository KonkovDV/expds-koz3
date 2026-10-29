"""Item 3: CRS / разрешение всех геофизических растров трейна (как их видит geo_pixgbm.load_map)."""
import sys, json, collections
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt')
import rasterio
from expds.catalog import scan_root, read_utm_json
from expds.crs import usable_raster_crs
from expds import geo_pixgbm as G
rows = []
for reg in scan_root('/workspace/data/train-expds-3'):
    if not (reg.name[:3].isdigit() and 200 <= int(reg.name[:3]) <= 217): continue
    utm = read_utm_json(reg.utm_json)
    for mt in ('maggr', 'etgr', 'terfl'):
        files = reg.rasters.get(mt, [])
        picked = set(G.pick_rasters(files))
        for f in files:
            with rasterio.open(f) as r:
                decl = r.crs.to_string() if r.crs else None
                used = usable_raster_crs(decl, r.bounds, utm)
                rows.append(dict(region=reg.name, mt=mt, file=f.name, picked=f in picked, declared=decl,
                                 used=used, utm_json=utm, res=round(abs(r.transform.a), 5),
                                 units=(r.crs.linear_units if r.crs else None)))
json.dump(rows, open('crs_inventory.json', 'w'), ensure_ascii=False, indent=1)
c = collections.Counter((r['mt'], r['declared'], r['used']) for r in rows)
for k, v in sorted(c.items(), key=str): print(k, v)
print('n', len(rows), 'picked', sum(r['picked'] for r in rows))
