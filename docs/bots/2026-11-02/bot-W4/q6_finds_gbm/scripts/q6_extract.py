#!/usr/bin/env python3
"""Q6 step 1: candidates + descriptors + labels per region -> cache/<region>.pkl (PREREG 870433c)."""
import sys, json, pickle, time, tomllib
sys.path.insert(0, '/workspace/koz3/w4/repo')
from pathlib import Path
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from expds import geophys, finds_cand as F
from expds.catalog import scan_root
from expds.pipeline import region_crs
from expds.raster import circle_3857
C = Path('/workspace/koz3/orch/bot-W4/work/q6/cache'); C.mkdir(exist_ok=True)
cfg = tomllib.load(open('/workspace/koz3/w4/repo/configs/v3_on.toml', 'rb'))
geo = cfg['geophys']
gt = json.loads(open('/workspace/koz3/w5data/gt_train_v2.geojson', 'rb').read())
GT = {}
for f in gt['features']:
    p = f['properties']
    if p['class_name'] == 'finds_points':
        GT.setdefault(p['region_name'], []).append(f['geometry']['coordinates'][:2])
t0 = time.time()
only = set(sys.argv[1:])
for reg in scan_root('/workspace/koz3/w4geo/train'):
    if only and reg.name not in only: continue
    files = geophys.select_rasters(reg.rasters.get('maggr', []), 'maggr', 'gray')
    if not files: continue
    out = C / f'{reg.name}.pkl'
    if out.exists(): continue
    crs_r = region_crs(reg, cfg)
    p, o = geophys.params_for(geo, 'maggr', 'finds_points')
    boundary = geophys.survey_boundary(reg.boundaries, 'maggr', crs_r, geo)
    noise = geophys.load_mask(reg.noise, 'maggr', crs_r)
    recs = []
    for path in files:
        b_map, n_map = geophys.sane_masks(path, boundary, noise, geo, crs_r)
        d = F.raster_candidates(path, p, crs_r)
        xy = d['xy']
        feats = [{'geom': circle_3857(float(x), float(y), 0.5, d['crs']), 'i': i} for i, (x, y) in enumerate(xy)]
        kept = geophys.apply_masks(feats, b_map, n_map if o['use_noise'] else None)
        km = np.zeros(len(xy), bool); km[[f['i'] for f in kept]] = True
        g = np.asarray(GT.get(reg.name, []), float).reshape(-1, 2)
        if len(g) and len(xy):
            tr = Transformer.from_crs(3857, d['crs'], always_xy=True)
            gx, gy = tr.transform(g[:, 0], g[:, 1])
            dist, _ = cKDTree(np.c_[gx, gy]).query(xy)
        else:
            dist = np.full(len(xy), np.inf)
        recs.append(dict(source=path.name, crs=d['crs'], res=d['res'], xy=xy, X=d['X'], dist=dist, kept=km))
        print(f'{reg.name} {path.name}: res {d["res"]:.3f} cand {len(xy)} kept {km.sum()} pos(<=1m) {(dist[km]<=1).sum()} '
              f'GT {len(g)} GT-covered {0 if not len(xy) else None} {time.time()-t0:.0f}s', flush=True)
    pickle.dump(recs, open(out, 'wb'))
print('done', f'{time.time()-t0:.0f}s')
