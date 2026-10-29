"""koz3-w5-dino-emb: frozen DINOv2 crop embeddings for stack candidates (features for the W5 reranker).
Input:  /kaggle/input/koz3-w5-cands/cands.csv  columns: cid,region,markup_type,tif(optional),cx,cy,w,h  (metres; optional column crs, default = raster CRS, i.e. the region UTM used by prep.py)
        raw rasters: /kaggle/input/<xfer datasets>/<region>/**.tif  (koz3-xfer-s* shards, unpacked)
Output: /kaggle/working/emb_<model>.npy (float16, N x D, row order = cands.csv), emb_ok.npy, meta.json
Device: CUDA if present, else XLA (TPU) if torch_xla present, else CPU (ViT-S/14 on CPU ~ 30-60 crops/s).
Crop: square window = max(32 m, 2.5 x max(w,h)) centred on (cx,cy), resampled to 224 px, 3 bands (1-band -> repeated)."""
import os, sys, json, glob, time, math, numpy as np, pandas as pd
import torch, rasterio
from rasterio.warp import transform as wtransform
from rasterio.windows import from_bounds
MODEL = os.environ.get('DINO', 'dinov2_vits14')          # dinov2_vits14 | dinov2_vitb14 (+ _reg)
INP = os.environ.get('CANDS', '/kaggle/input/koz3-w5-cands/cands.csv')
RAW = os.environ.get('RAW', '/kaggle/input')
OUT = os.environ.get('OUT', '/kaggle/working'); BS = int(os.environ.get('BS', 128)); S = 224
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    if dev == 'cpu' and (os.environ.get('TPU_NAME') or os.path.exists('/dev/accel0')):
        import torch_xla.core.xla_model as xm; dev = xm.xla_device()
except Exception: pass
# weights: offline dataset (facebookresearch/dinov2 hub dir) if attached, else torch.hub (needs Internet ON)
hub = glob.glob('/kaggle/input/*/dinov2*/hubconf.py')
m = torch.hub.load(os.path.dirname(hub[0]), MODEL, source='local') if hub else torch.hub.load('facebookresearch/dinov2', MODEL)
m = m.eval().to(dev).half() if dev == 'cuda' else m.eval().to(dev)
C = pd.read_csv(INP); N = len(C); print(N, 'cands', dev, MODEL, flush=True)
tifs = {}
for p in glob.glob(f'{RAW}/**/*.tif', recursive=True):
    reg = next((x for x in p.split('/') if x[:3].isdigit() and '_' in x), None)
    if reg: tifs.setdefault(reg, []).append(p)
def pick(row):
    L = tifs.get(row.region, [])
    if isinstance(row.get('tif'), str): L = [p for p in L if p.endswith(row.tif)] or L
    mt = str(row.markup_type).lower()
    tag = {'ae': '_ae', 'li': '_li', 'sp': '_sp', 'or': '_or'}.get(mt[:2], '')
    return ([p for p in L if tag and tag in os.path.basename(p).lower()] or L or [None])[0]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
def crop(ds, x, y, half, src=None):
    xs, ys = ([x - half, x + half], [y - half, y + half]) if src is None else wtransform(src, ds.crs, [x - half, x + half], [y - half, y + half])
    win = from_bounds(min(xs), min(ys), max(xs), max(ys), ds.transform)
    a = ds.read(indexes=list(range(1, min(ds.count, 3) + 1)), window=win, out_shape=(min(ds.count, 3), S, S), boundless=True, fill_value=0,
                resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
    if a.shape[0] == 1: a = np.repeat(a, 3, 0)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99); a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a - MEAN) / STD
D = m.embed_dim; E = np.zeros((N, 2 * D), np.float16); ok = np.zeros(N, bool); t0 = time.time()
C['tifp'] = [pick(r) for _, r in C.iterrows()]
for tp, g in C.groupby('tifp', dropna=True):
    with rasterio.open(tp) as ds:
        idx = g.index.to_numpy()
        for s in range(0, len(idx), BS):
            ii = idx[s:s + BS]; X = []
            for i in ii:
                r = C.loc[i]; half = max(16.0, 1.25 * max(r.w, r.h)); X.append(crop(ds, r.cx, r.cy, half, r.get('crs') if isinstance(r.get('crs'), str) else None))
            xb = torch.from_numpy(np.stack(X)).to(dev); xb = xb.half() if dev == 'cuda' else xb
            with torch.no_grad():
                f = m.forward_features(xb); cls_t = f['x_norm_clstoken']; pat = f['x_norm_patchtokens'].mean(1)
            E[ii] = torch.cat([cls_t, pat], 1).float().cpu().numpy().astype(np.float16); ok[ii] = True
    print(f'{ok.sum()}/{N} {time.time()-t0:.0f}s', flush=True)
np.save(f'{OUT}/emb_{MODEL}.npy', E); np.save(f'{OUT}/emb_ok.npy', ok)
json.dump({'model': MODEL, 'n': int(N), 'ok': int(ok.sum()), 'sec': round(time.time() - t0), 'dev': str(dev)}, open(f'{OUT}/meta.json', 'w'))
