#!/usr/bin/env python3
"""KOZ3 sat2 / sp_det2 fold0 pilot (PROMPT_SAT2_DETECTOR_20261101, A lease slot:dmitrijkv/gpu-1).
CenterNet (in-repo centerhead/) on timm ConvNeXt-T U-Net (ImageNet), 3-ch SpOr, kurgans only, stride 4, res 0.5 m, 512 px.
Train = GRID cache (expds.tiles --grid-overlap 0.25) of pinned spor_focal folds 1–4 (framing tiles; neg cap 2×pos/region),
site-first sampler, focal heatmap loss, D8 + quality aug. Test = full fold0 («Курганы», 34 regions) grid → OOF geojson
(no-TTA and D8-TTA), own-cell dedupe across overlapping windows. No credentials, no S3, no LB. Fixed schedule (no fold0 peeking)."""
from __future__ import annotations
import os, sys, subprocess, time, json, csv, math, importlib, importlib.metadata
from pathlib import Path
T0 = time.time()
SMOKE = os.environ.get('SAT2_SMOKE') == '1'
DEADLINE_H = float(os.environ.get('SAT2_DEADLINE_H', '7.5'))
EPOCHS = int(os.environ.get('SAT2_EPOCHS', '40')); SPE = int(os.environ.get('SAT2_SPE', '1600'))
BATCH = int(os.environ.get('SAT2_BATCH', '8')); LR = float(os.environ.get('SAT2_LR', '3e-4'))
ENC = os.environ.get('SAT2_ENC', 'tu-convnext_tiny')
OUT = Path(os.environ.get('SAT2_OUT', '/kaggle/working')); OUT.mkdir(parents=True, exist_ok=True)
INP = Path(os.environ.get('SAT2_INPUT', '/kaggle/input'))

def log(*a):
    print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
def status(s):
    (OUT / 'STATUS.txt').write_text(s + '\n'); log('STATUS', s)
def left_h():
    return DEADLINE_H - (time.time() - T0) / 3600
status('RUNNING')
if not SMOKE:
    smi = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout
    if 'GPU' not in smi:
        status('FAILED no GPU'); raise SystemExit('no GPU')
    log('gpu', smi.strip())
    import torch  # noqa
    cons = '/tmp/cons.txt'
    Path(cons).write_text(''.join(f'{p}=={importlib.metadata.version(p)}\n' for p in ('torch', 'torchvision', 'numpy')))
    need = []
    for mod, pin in {'segmentation_models_pytorch': 'segmentation-models-pytorch==0.6.0', 'timm': 'timm==1.0.29',
                     'pyproj': 'pyproj', 'shapely': 'shapely', 'scipy': 'scipy'}.items():
        try:
            importlib.import_module(mod)
        except Exception:
            need.append(pin)
    if need:
        log('pip', need)
        if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-c', cons, *need]).returncode:
            status('FAILED pip'); raise SystemExit(1)

def find(name):
    for p in sorted(INP.rglob(name)):
        return p
    raise FileNotFoundError(name)
code = find('centerhead').parent
sys.path[:0] = [str(code), str(code / 'sat2_code')]
pack = (find('pack.json').parent).parent
train_dir, test_dir = pack / 'train', pack / 'f0'
log('code', code, 'pack', pack)

import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from centerhead.config import cfg_with, HM_CLASSES
from centerhead.head import build_unet_centernet
from centerhead.losses import center_loss
from centerhead.decode import decode, concat
from centerhead.d8 import torch_untta_offsets
from centerhead.export import candidates_to_feats, to_fc
from grid_dataset import GridSplit, CenterGridDataset, site_first_weights, IMNET_MEAN, IMNET_STD
CFG = cfg_with(markup_type='sp_or', circle_r_m=4.0, min_score=0.03)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); np.random.seed(0)
tr = GridSplit(train_dir)
log('train tiles', len(tr), 'pos', sum(int(r['n_obj']) > 0 for r in tr.rows), 'folds', sorted({r['fold'] for r in tr.rows}))
assert '0' not in {r['fold'] for r in tr.rows}, 'fold0 leak'
if SMOKE:
    EPOCHS, SPE, BATCH = 1, 4, 2
ds = CenterGridDataset(tr, True, CFG)
w = site_first_weights(tr, neg_frac=0.3)
model = build_unet_centernet(3, ENC, 'imagenet', CFG).to(dev)
log('model', ENC, round(sum(p.numel() for p in model.parameters()) / 1e6, 1), 'M')
enc_p = list(model.encoder.parameters()); ids = {id(p) for p in enc_p}
opt = torch.optim.AdamW([{'params': enc_p, 'lr': LR * 0.5}, {'params': [p for p in model.parameters() if id(p) not in ids], 'lr': LR}],
                        weight_decay=1e-4)
steps_total = EPOCHS * (SPE // BATCH); warm = max(1, steps_total // 40)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps_total))))
scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda')
hist = []; train_budget_h = DEADLINE_H - 1.0
for ep in range(EPOCHS):
    ds.epoch = ep
    dl = DataLoader(ds, BATCH, sampler=WeightedRandomSampler(torch.tensor(w), SPE, True, torch.Generator().manual_seed(ep)),
                    num_workers=0 if SMOKE else 4, drop_last=True, pin_memory=dev == 'cuda', persistent_workers=False)
    model.train(); acc = {}; nb = 0; te = time.time()
    for b in dl:
        x = b['image'].to(dev, non_blocking=True)
        batch = {k: b[k].to(dev, non_blocking=True) for k in ('hm', 'wh', 'off', 'reg_mask')}
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
            out = model(x)
        out = {k: (v.float() if v is not None else None) for k, v in out.items() if k != 'mask'}
        L = center_loss(out, batch, CFG)
        opt.zero_grad(set_to_none=True); scaler.scale(L['loss_center']).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step(); nb += 1
        for k, v in L.items():
            acc[k] = acc.get(k, 0) + float(v.detach() if torch.is_tensor(v) else v)
    h = {'epoch': ep, **{k: round(v / max(nb, 1), 5) for k, v in acc.items()}, 'sec': round(time.time() - te, 1)}
    hist.append(h); log('ep', h)
    if not all(math.isfinite(v) for k, v in h.items() if k.startswith('loss')):
        status(f'FAILED nan ep{ep}'); raise SystemExit(1)
    if (time.time() - T0) / 3600 > train_budget_h:
        log('stop train: budget'); break
torch.save({'state_dict': model.state_dict(), 'enc': ENC, 'cfg': {k: v for k, v in CFG.items() if not isinstance(v, tuple)},
            'hm_classes': list(HM_CLASSES), 'epochs_done': len(hist)}, OUT / 'sp_det2_cnxt_f0.pt')
json.dump(hist, open(OUT / 'train_hist.json', 'w'), indent=1)

# ---------------- fold0 grid inference
te = GridSplit(test_dir, mmap=True)
assert {r['fold'] for r in te.rows} == {'0'}, 'test must be fold0 only'
rows = te.rows; S = te.S; st = CFG['stride']
if SMOKE:
    rows = rows[:4]
from scipy.spatial import cKDTree
src_idx = {}
for i, r in enumerate(rows):
    src_idx.setdefault((r['region'], r['file']), []).append(i)
trees = {k: (cKDTree(np.array([[float(rows[i]['cx']), float(rows[i]['cy'])] for i in v]), ), v) for k, v in src_idx.items()}
model.eval()
VIEWS = {'notta': [(0, False)], 'd8': [(k, f) for k in range(4) for f in (False, True)]}
cands = {v: [] for v in VIEWS}
mean = torch.tensor(IMNET_MEAN, device=dev); std = torch.tensor(IMNET_STD, device=dev)
IB = 1 if SMOKE else 8
with torch.no_grad():
    for b0 in range(0, len(rows), IB):
        ids = list(range(b0, min(len(rows), b0 + IB)))
        x = (torch.from_numpy(np.stack([np.asarray(te.images[i]) for i in ids])).to(dev).float() - mean) / std
        agg = {}
        for k, fl in VIEWS['d8']:
            xv = torch.rot90(x, k, (2, 3)); xv = xv.flip(3) if fl else xv
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
                o = model(xv)
            hm = torch.sigmoid(o['hm'].float()); wh = o['wh'].float()
            if fl:
                hm, wh = hm.flip(3), wh.flip(3)
            hm, wh = torch.rot90(hm, -k, (2, 3)), torch.rot90(wh, -k, (2, 3))
            off = torch_untta_offsets(o['off'].float(), k, fl)
            if (k, fl) == (0, False):
                agg['notta'] = (hm.cpu().numpy(), wh.cpu().numpy(), off.cpu().numpy())
            a = agg.get('d8_sum')
            agg['d8_sum'] = (hm, wh, off) if a is None else tuple(p + q for p, q in zip(a, (hm, wh, off)))
        agg['d8'] = tuple((v / 8).cpu().numpy() for v in agg.pop('d8_sum'))
        for v in VIEWS:
            hm, wh, off = agg[v]
            for j, i in enumerate(ids):
                r = rows[i]; step = float(r['step'])
                t = decode(hm[j], float(r['left']), float(r['top']), step * st, wh[j], off[j], cfg=CFG)
                if len(t['X']):
                    tree, members = trees[(r['region'], r['file'])]
                    _, nn = tree.query(np.stack([t['X'], t['Y']], 1), p=np.inf)
                    keep = np.asarray(members)[nn] == i
                    t = {kk: np.asarray(vv)[keep] for kk, vv in t.items()}
                    t['crs'] = np.array([r['crs']] * len(t['X'])); t['region'] = np.array([r['region']] * len(t['X']))
                    cands[v].append(t)
        if b0 % 200 == 0:
            log('infer', b0, '/', len(rows))
summ = {'epochs_done': len(hist), 'train_tiles': len(tr), 'test_tiles': len(rows), 'enc': ENC, 'last_loss': hist[-1] if hist else None}
for v, tl in cands.items():
    feats = []
    if tl:
        allc = concat(tl)
        for (reg, crs) in sorted({(a, b) for a, b in zip(allc['region'], allc['crs'])}):
            m = (allc['region'] == reg) & (allc['crs'] == crs)
            feats += candidates_to_feats({kk: vv[m] for kk, vv in allc.items()}, str(reg), str(crs), cfg=CFG)
    fc = to_fc(feats)
    for f in fc['features']:
        f['properties']['threshold_group'] = 'sp_det2'
    json.dump(fc, open(OUT / f'sp_det2_fold0_{v}.geojson', 'w'))
    summ[f'n_{v}'] = len(fc['features'])
json.dump(summ, open(OUT / 'summary.json', 'w'), indent=1)
status(f'DONE {time.time()-T0:.0f}s')
