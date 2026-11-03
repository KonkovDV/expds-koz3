#!/usr/bin/env python3
"""CPU smoke: centerhead UnetCenterNet (timm ConvNeXt-T, fallback R50) on 4 GRID tiles (512 px, overlap 0.25) of a fold0 SpOr tif.
Checks: build+ImageNet load, targets from gt_train_v2 (sp_or kurgans), 3 train steps finite, decode → candidates → 3857 feats."""
import json, sys, time, math
from pathlib import Path
import numpy as np, torch, rasterio
from rasterio.windows import Window
sys.path.insert(0, '/workspace/koz3/wt/sp-large-f0')
from centerhead import DEFAULTS
from centerhead.head import build_unet_centernet
from centerhead.geo import gt_to_objs, crop_objs
from centerhead.targets import render_targets_strided
from centerhead.losses import center_loss
from centerhead.decode import decode, tile_origins
from centerhead.export import candidates_to_feats
K = Path('/workspace/koz3'); OUT = K / 'orch/bot-W1/sat2_smoke'; T0 = time.time(); res = {}
REG = sys.argv[1] if len(sys.argv) > 1 else '130_Курганы_1'
tif = sorted((K / 'w5data/sat_raw' / REG).rglob('*SpOr*.tif'))[0]
gt = [f for f in json.load(open(K / 'w5data/gt_train_v2.geojson'))['features']
      if f['properties']['region_name'] == REG and f['properties']['markup_type'] == 'sp_or'
      and f['properties']['class_name'] in ('kurgany_tselye', 'kurgany_povrezhdennye')]
S, STEP = 512, int(512 * (1 - 0.25))
with rasterio.open(tif) as ds:
    crs = ds.crs.to_string(); tr = ds.transform; H, W = ds.height, ds.width; step = abs(tr.a)
    objs = gt_to_objs(gt, tr.c, tr.f, step, 'EPSG:3857', crs, cfg=DEFAULTS, markup_type='sp_or')
    origins = [tuple(map(int, o)) for o in tile_origins(H, W, S, STEP, DEFAULTS['stride'])]  # grid incl. edge-aligned last tile
    cnt = [((objs[:, 0] >= x) & (objs[:, 0] < x + S) & (objs[:, 1] >= y) & (objs[:, 1] < y + S)).sum() for y, x in origins]
    pick = [origins[i] for i in np.argsort(cnt)[::-1][:4]]
    tiles = [ds.read([1, 2, 3], window=Window(x, y, S, S), boundless=True, fill_value=0) for y, x in pick]
res.update(region=REG, tif=str(tif.relative_to(K)), crs=crs, res_m=round(step, 4), raster_hw=[H, W], n_gt_kurgans=len(gt),
           grid_windows=len(origins), picked=[list(map(int, p)) for p in pick], gt_in_picked=[int(cnt[origins.index(p)]) for p in pick])
x = torch.from_numpy((np.stack(tiles).astype(np.float32) - 128.0) / 127.0)
tg = [render_targets_strided(crop_objs(objs, px, py, S), S, DEFAULTS['stride'], 2, DEFAULTS) for py, px in pick]
batch = {k: torch.from_numpy(np.stack([t[k] for t in tg])) for k in ('hm', 'wh', 'off', 'reg_mask')}
model = None
for enc in ('tu-convnext_tiny', 'tu-resnet50'):
    for w in ('imagenet', None):
        try:
            model = build_unet_centernet(3, enc, w); res.update(encoder=enc, weights=w); break
        except Exception as e:
            res.setdefault('load_errors', []).append(f'{enc}/{w}: {type(e).__name__}: {str(e)[:160]}')
    if model is not None: break
res['params_M'] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
opt = torch.optim.AdamW(model.parameters(), 1e-4); losses = []
model.train()
for it in range(3):
    out = model(x); L = center_loss(out, batch, DEFAULTS)
    opt.zero_grad(); L['loss_center'].backward(); opt.step()
    losses.append({k: round(float(v), 5) for k, v in L.items()})
res['train_losses'] = losses
res['out_shapes'] = {k: list(v.shape) for k, v in out.items() if v is not None}
model.eval()
with torch.no_grad(): o = model(x[:1])
hm = torch.sigmoid(o['hm'][0]).numpy(); py, px = pick[0]; st = step * DEFAULTS['stride']
cand = decode(hm, tr.c + px * step, tr.f - py * step, st, o['wh'][0].numpy(), o['off'][0].numpy(), cfg=DEFAULTS)
feats = candidates_to_feats(cand, REG, crs, cfg=dict(DEFAULTS, markup_type='sp_or'))
res.update(n_candidates=len(cand['X']), n_feats=len(feats), feat0={k: v for k, v in (feats[0].items() if feats else []) if k != 'geom'},
           hm_max=float(hm.max()))
fin = all(math.isfinite(l['loss_center']) for l in losses)
res['PASS'] = bool(fin and res['out_shapes']['hm'][1:] == [2, S // 4, S // 4] and sum(res['gt_in_picked']) > 0 and 'encoder' in res)
res['elapsed_s'] = round(time.time() - T0, 1)
json.dump(res, open(OUT / 'smoke_result.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
