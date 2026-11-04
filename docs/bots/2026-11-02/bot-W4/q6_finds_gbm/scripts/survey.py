import sys, json
sys.path.insert(0, '/workspace/koz3/w4/repo')
from pathlib import Path
import numpy as np, rasterio
from expds import geophys
from expds.catalog import scan_root
from scipy.spatial import cKDTree
from pyproj import Transformer
gt = json.loads(open('/workspace/koz3/w5data/gt_train_v2.geojson','rb').read())
G = {}
for f in gt['features']:
    p = f['properties']
    if p['class_name'] == 'finds_points':
        c = f['geometry']['coordinates']
        while isinstance(c[0], list): c = c[0]
        G.setdefault(p['region_name'], []).append(c[:2] if f['geometry']['type']=='Point' else None)
print('gt geom types', {f['geometry']['type'] for f in gt['features'] if f['properties']['class_name']=='finds_points'})
for reg in scan_root('/workspace/koz3/w4geo/train'):
    files = geophys.select_rasters(reg.rasters.get('maggr', []), 'maggr', 'gray')
    if not files: continue
    for fp in files:
        with rasterio.open(fp) as ds:
            print(reg.name, fp.name, ds.width, ds.height, ds.count, ds.dtypes[0], ds.res, ds.crs, flush=True)
    pts = [p for p in G.get(reg.name, []) if p]
    if len(pts) > 1:
        a = np.array(pts); 
        # gt likely 3857 -> metres approx scale
        d, _ = cKDTree(a).query(a, 2); print('  gt', len(a), 'NN dist pct (units)', np.percentile(d[:,1],[5,25,50]).round(2), a[0])
