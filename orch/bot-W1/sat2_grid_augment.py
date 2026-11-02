#!/usr/bin/env python3
"""Augment sat2_dataset_manifest.json for GRID tiles (expds.tiles --grid-overlap 0.25): per-tif bounds/res and
estimated grid windows at res 0.5 m / tile 512 (256 m, centerhead default) and positive windows (sp_or kurgan centroid inside).
Also checks whether family_folds weighted by GRID positives would change folds (we PIN spor_focal folds for comparability)."""
import json, math, collections
from pathlib import Path
import numpy as np, rasterio
from pyproj import Transformer
from shapely.geometry import shape
K = Path('/workspace/koz3'); M = K / 'orch/bot-W1/sat2_dataset_manifest.json'
man = json.load(open(M)); RES, TILE, OV = 0.5, 512, 0.25; SPAN = RES * TILE; STEP = SPAN * (1 - OV)
gt = collections.defaultdict(list)
for f in json.load(open(K / 'w5data/gt_train_v2.geojson'))['features']:
    p = f['properties']
    if p['markup_type'] == 'sp_or' and p['class_name'].startswith('kurgany') and f.get('geometry'):
        c = shape(f['geometry']).centroid; gt[p['region_name']].append((c.x, c.y))
def axis(lo, hi):
    n = max(1, math.ceil((hi - lo - SPAN) / STEP) + 1) if hi - lo > SPAN else 1
    return [lo + SPAN / 2 + i * STEP for i in range(n)]
tot = collections.Counter(); posfam = collections.Counter()
for r in man['regions']:
    r['grid'] = []
    for t in r['tifs']:
        with rasterio.open(K / t) as ds:
            crs = ds.crs.to_string(); b = ds.bounds; res = abs(ds.transform.a)
        tf = Transformer.from_crs('EPSG:3857', crs, always_xy=True)
        pts = np.array([tf.transform(x, y) for x, y in gt.get(r['region'], [])]).reshape(-1, 2)
        k = 1.0
        if crs.endswith('3857'): k = 1 / math.cos(math.atan(math.sinh(((b.bottom + b.top) / 2) / 6378137.0)))  # map m per ground m
        xs = axis(b.left, b.right) if k == 1 else [b.left + (SPAN * k) / 2 + i * STEP * k for i in range(max(1, math.ceil(((b.right - b.left) / k - SPAN) / STEP) + 1))]
        ys = axis(b.bottom, b.top) if k == 1 else [b.bottom + (SPAN * k) / 2 + i * STEP * k for i in range(max(1, math.ceil(((b.top - b.bottom) / k - SPAN) / STEP) + 1))]
        half = SPAN * k / 2; n = len(xs) * len(ys); pos = 0
        if len(pts):
            for cx in xs:
                for cy in ys:
                    pos += bool(((abs(pts[:, 0] - cx) < half) & (abs(pts[:, 1] - cy) < half)).any())
        r['grid'].append({'tif': t, 'crs': crs, 'native_res_m': round(res * (1 / k if k != 1 else 1), 4), 'grid_windows': n, 'grid_pos_windows': pos})
        tot['windows'] += n; tot['pos'] += pos; posfam[r['family']] += pos
# fold stability under grid weights
fams = sorted({r['family'] for r in man['regions'] if r['fold'] is not None})
load = [0] * 5; gfold = {}
for f in sorted(fams, key=lambda f: (-posfam[f], f)):
    i = min(range(5), key=lambda j: load[j]); gfold[f] = i; load[i] += posfam[f]
changed = sorted(f for f in fams if gfold[f] != next(r['fold'] for r in man['regions'] if r['family'] == f))
fold_tiles = {str(k): sum(g['grid_windows'] for r in man['regions'] if r['fold'] == k for g in r['grid']) for k in range(5)}
fold_pos = {str(k): sum(g['grid_pos_windows'] for r in man['regions'] if r['fold'] == k for g in r['grid']) for k in range(5)}
man['grid_spec'] = {'cmd': 'python -m expds.tiles ... --markup sp_or --res 0.5 --tile 512 --grid-overlap 0.25 (classes kurgany_povrezhdennye kurgany_tselye)',
                    'res_m': RES, 'tile_px': TILE, 'overlap': OV, 'note': 'window counts are estimates from tif bounds; positive = sp_or kurgan centroid inside window'}
man['summary']['grid'] = {'windows_total': tot['windows'], 'pos_windows_total': tot['pos'], 'windows_per_fold': fold_tiles, 'pos_windows_per_fold': fold_pos,
                          'est_cache_GB_uint8_3ch': round(tot['windows'] * TILE * TILE * 3 / 1e9, 2),
                          'families_whose_fold_changes_if_weighted_by_grid_pos': changed,
                          'fold_policy': 'PIN folds from spor_focal recipe (nn_oof family_folds on cache_spor positives) — needed for fold0 vs spor_focal baseline; nn_oof on a grid cache would recompute folds from grid positives → pass explicit split'}
man['summary']['native_res_m_range'] = [min(g['native_res_m'] for r in man['regions'] for g in r['grid']), max(g['native_res_m'] for r in man['regions'] for g in r['grid'])]
json.dump(man, open(M, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(man['summary']['grid'], ensure_ascii=False, indent=1), man['summary']['native_res_m_range'])
