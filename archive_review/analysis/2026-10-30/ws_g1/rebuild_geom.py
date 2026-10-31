"""Rebuild work/geom_wkb.pkl + gt_wkb.pkl (UTM geoms of table.pkl rows) exactly like prep.py, geometry only; asserts cls/conf alignment."""
import sys, os; sys.path.insert(0, os.getcwd())
from common import *
import numpy as np, shapely, ijson
from expds import metric as M
mod = M.load_official(METRIC); calc = mod.F1Calculator()
M._SHM_CACHE_DIR = Path('/nonexistent_ro/x')
T = pickle.load(open(W/'work/table.pkl', 'rb')); CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}
geom = []; n = 0
def flush(chunk):
    global n
    pr = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': chunk}, False); M._UTM_GEOM_CACHE.clear()
    it = {c: iter(v) for c, v in pr.items()}
    for f in chunk:
        c = mod.normalize_class_name(f['properties'].get('class_name', 'unknown'))
        assert cid.get(c, -1) == T['cls'][n], (n, c)
        geom.append(next(it[c])[0] if c in cid else None); n += 1
chunk = []
with open(W/STACK, 'rb') as fh:
    for f in ijson.items(fh, 'features.item', use_float=True):
        chunk.append(f)
        if len(chunk) == 50000: flush(chunk); chunk = []; print(n, flush=True)
flush(chunk); assert n == len(T['cls'])
g = np.array(geom, dtype=object); c = shapely.centroid(g)
print('max centroid diff', float(np.nanmax(np.abs(shapely.get_x(c) - T['cx']))), flush=True)
pickle.dump(shapely.to_wkb(g), open(W/'work/geom_wkb.pkl', 'wb'), protocol=5)
gt = load(W/'gt_train_v2.geojson'); gts = M._collect(mod, calc, gt, True)
G = {c: [x[0] for x in gts.get(c, [])] for c in CL}
for c in CL: assert len(G[c]) == T['n_gt'][c]
pickle.dump({c: shapely.to_wkb(G[c]) for c in CL}, open(W/'work/gt_wkb.pkl', 'wb'), protocol=5)
print('done')
