#!/usr/bin/env python3
"""CPU: GRID cache (expds.tiles --grid-overlap) → centerhead labels per tile, PINNED folds from sat2_dataset_manifest.json.
Writes into each cache dir:
  tiles.csv  : shard,idx,region,family,file,fold,crs,step,left,top,n_obj,n_povr,n_tsel,framing
  objs.npy   : float32 (M,5) [tile_row, x_px, y_px, diam_px, cls]  — cls in centerhead HM_CLASSES order (0 tselye, 1 povr)
Pixel coords are of the 512 input crop (x=(X-left)/step, y=(top-Y)/step), objects with centre up to D outside kept (tails).
framing=1: tile centre within FRAME_M of any sp_or GT object (any class) of that region (PROMPT: no far negatives).
Folds are NOT recomputed from grid positives (spor_focal family folds pinned)."""
import csv, json, sys, math
from pathlib import Path
import numpy as np
K = Path('/workspace/koz3'); sys.path.insert(0, str(K / 'wt/sp-large-f0'))
from shapely.geometry import shape
from expds.raster import open_metric
from expds.tiles import working_res
from centerhead.config import DEFAULTS, HM_CLASSES
from centerhead.geo import gt_to_objs, crop_objs, transformer
FRAME_M = 384.0
man = json.load(open(K / 'orch/bot-W1/sat2_dataset_manifest.json'))
FOLD = {r['region']: r['fold'] for r in man['regions']}
gt = json.load(open(K / 'w5data/gt_train_v2.geojson'))['features']
by_reg = {}
for f in gt:
    p = f['properties']
    if p.get('markup_type') == 'sp_or' and f.get('geometry'):
        by_reg.setdefault(p['region_name'], []).append(f)

def run(cache: Path, root: Path):
    meta = json.loads((cache / 'meta.json').read_text()); S = int(meta['tile']); res = float(meta['res'])
    rows = list(csv.DictReader(open(cache / 'index.csv', encoding='utf-8')))
    step_cache, objs_cache, frame_cache = {}, {}, {}
    out_rows, objs_all = [], []
    for i, r in enumerate(rows):
        key = f"{r['region']}|{r['file']}"; crs = meta['crs_by_source'][key]
        if key not in step_cache:
            tif = next((root / r['region']).rglob(r['file']))
            with open_metric(tif) as ds:
                step_cache[key] = working_res(abs(ds.res[0]), res)
            feats = by_reg.get(r['region'], [])
            kf = [f for f in feats if f['properties']['class_name'] in HM_CLASSES]
            objs_cache[key] = gt_to_objs(kf, 0.0, 0.0, 1.0, 'EPSG:3857', crs, HM_CLASSES, DEFAULTS, 'sp_or')  # metres: x=X, y=-Y
            t = transformer('EPSG:3857', crs)
            pts = np.array([t.transform(*shape(f['geometry']).representative_point().coords[0]) for f in feats]).reshape(-1, 2)
            frame_cache[key] = pts
        st = step_cache[key]; cx, cy = float(r['cx']), float(r['cy'])
        left, top = cx - S * st / 2, cy + S * st / 2
        om = objs_cache[key].copy()
        if len(om):   # from metres (X, -Y, D) to crop pixels
            om[:, 0] = (om[:, 0] - left) / st; om[:, 1] = (top + om[:, 1]) / st; om[:, 2] = om[:, 2] / st
        o = crop_objs(om, 0, 0, S)
        inside = (o[:, 0] >= 0) & (o[:, 0] < S) & (o[:, 1] >= 0) & (o[:, 1] < S)
        pts = frame_cache[key]
        fr = int(bool(len(pts)) and bool((np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < FRAME_M + S * st / 2).any()))
        for ob in o:
            objs_all.append([len(out_rows), *ob])
        out_rows.append({**{k: r[k] for k in ('shard', 'idx', 'region', 'family', 'file')}, 'fold': FOLD.get(r['region']),
                         'crs': crs, 'step': round(st, 6), 'left': round(left, 3), 'top': round(top, 3), 'cx': cx, 'cy': cy,
                         'n_obj': int(inside.sum()), 'n_tsel': int((o[inside, 3] == 0).sum()), 'n_povr': int((o[inside, 3] == 1).sum()),
                         'framing': fr})
    with open(cache / 'tiles.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    np.save(cache / 'objs.npy', np.asarray(objs_all, np.float32).reshape(-1, 5))
    s = {'cache': str(cache), 'tiles': len(out_rows), 'pos': sum(r['n_obj'] > 0 for r in out_rows),
         'framing': sum(r['framing'] for r in out_rows), 'objs_inside': sum(r['n_obj'] for r in out_rows),
         'folds': sorted({str(r['fold']) for r in out_rows})}
    print(json.dumps(s, ensure_ascii=False)); return s

if __name__ == '__main__':
    root = K / 'w5data/sat_raw'
    for c in sys.argv[1:]:
        run(Path(c), root)
