"""table.pkl -> cands.csv (+ .zst) for koz3-w5-dino-emb. cid = row in table.
prep stores centres in the per-geometry UTM zone of the official metric (zone from own centroid); recover the zone
(unique z with inverse-lon inside zone z) and export EPSG:3857 centres + size in metres. Optional --mask npy (bool) limits rows."""
from common import *
import numpy as np, shapely, shapely.geometry, csv, argparse
from pyproj import Transformer
ap = argparse.ArgumentParser(); ap.add_argument('--mask', default=''); ap.add_argument('--out', default='cands.csv'); a = ap.parse_args()
T = pickle.load(open(W/'work/table.pkl', 'rb')); g = shapely.from_wkb(pickle.load(open(W/'work/geom_wkb.pkl', 'rb')))
b = shapely.bounds(g); cx, cy = T['cx'], T['cy']; N = len(cx)
X3857 = np.full(N, np.nan); Y3857 = np.full(N, np.nan); zone = np.zeros(N, int)
# UTM easting/northing alone do not identify the zone -> candidate zones per region from GT (3857 -> lon), pick the one landing in the region's GT bbox (+20 km)
gt = load(W/'gt_train_v2.geojson'); to_wgs = Transformer.from_crs(3857, 4326, always_xy=True); RZ = {}; RB = {}
for f in gt['features']:
    r = (f.get('properties') or {}).get('region_name'); gm = f.get('geometry')
    if not r or not gm: continue
    c = shapely.geometry.shape(gm).centroid; lon, lat = to_wgs.transform(c.x, c.y)
    RZ.setdefault(r, set()).add(int((lon + 180) // 6) + 1); bb = RB.setdefault(r, [c.x, c.y, c.x, c.y])
    bb[0], bb[1], bb[2], bb[3] = min(bb[0], c.x), min(bb[1], c.y), max(bb[2], c.x), max(bb[3], c.y)
del gt
regs = T['reg']
for r in np.unique(regs):
    ii = np.flatnonzero(regs == r); zs = sorted(RZ.get(r, ())) or [37]
    bb = RB.get(r, [-1e12, -1e12, 1e12, 1e12])
    for z in zs:
        jj = ii[zone[ii] == 0]
        if not len(jj): break
        x, y = Transformer.from_crs(32600 + z, 3857, always_xy=True).transform(cx[jj], cy[jj])
        ok = (x > bb[0] - 2e4) & (x < bb[2] + 2e4) & (y > bb[1] - 2e4) & (y < bb[3] + 2e4) if len(zs) > 1 or r in RB else np.ones(len(jj), bool)
        zone[jj[ok]] = z; X3857[jj[ok]] = x[ok]; Y3857[jj[ok]] = y[ok]
print('zones', dict(zip(*np.unique(zone, return_counts=True))), 'unresolved', int((zone == 0).sum()))
m = np.load(a.mask) if a.mask else np.ones(N, bool)
with open(W/'work'/a.out, 'w', newline='', encoding='utf-8') as f:
    wr = csv.writer(f); wr.writerow(['cid', 'region', 'markup_type', 'cls', 'cx', 'cy', 'crs', 'w', 'h'])
    for i in np.flatnonzero(m & (zone > 0)):
        wr.writerow([i, T['reg'][i], T['mk'][i], T['CL'][T['cls'][i]] if T['cls'][i] >= 0 else '', f"{X3857[i]:.2f}", f"{Y3857[i]:.2f}", 'EPSG:3857',
                     f"{b[i,2]-b[i,0]:.2f}", f"{b[i,3]-b[i,1]:.2f}"])
print('rows', int((m & (zone > 0)).sum()))
