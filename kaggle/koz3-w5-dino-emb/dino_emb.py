"""koz3-w5-dino-emb: frozen DINOv2 crop embeddings for W5 reranker pool candidates.
Input : /kaggle/input/**/cands.csv.zst (cid,region,markup_type,cls,cx,cy,crs=EPSG:3857,w,h; w,h metres, UTM of region)
        li  : /kaggle/input/**/cnx_relief/<region>.npy (uint8 [lrm_s,lrm_l,slope,svf_r,opp_r] 0.5 m) + .json (crs,left,top,step)
              fallback / other layers: S3 bucket train-expds-3 rasters read by /vsis3 windows (keys from cred.json; never printed)
Output: /kaggle/working/emb.npy (float16 N x 2D: CLS + mean patch), ok.npy, cid.npy, meta.json; partial saves every 20 min.
Device: CUDA (half) > torch_xla (TPU) > CPU."""
import os, sys, io, json, glob, time, subprocess, numpy as np
def pip(*p): subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *p], check=False)
try: import zstandard
except ImportError: pip('zstandard'); import zstandard
try: import rasterio
except ImportError: pip('rasterio'); import rasterio
import csv, torch
from rasterio.warp import transform as wtransform
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
MODEL = os.environ.get('DINO', 'dinov2_vits14'); OUT = '/kaggle/working'; BS = int(os.environ.get('BS', 64)); S = 224
DEADLINE = time.time() + float(os.environ.get('DEADLINE_H', 11.3)) * 3600; T0 = time.time()
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
# ---- cands
p = glob.glob('/kaggle/input/**/cands.csv.zst', recursive=True)[0]
rows = list(csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(p, 'rb')), encoding='utf-8')))
N = len(rows); log(N, 'cands from', p)
# ---- device / model
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu' and (os.environ.get('TPU_NAME') or os.path.exists('/dev/accel0') or os.environ.get('PJRT_DEVICE') == 'TPU'):
    try: import torch_xla.core.xla_model as xm; dev = xm.xla_device()
    except Exception as e: log('xla unavailable', type(e).__name__)
if dev == 'cpu': torch.set_num_threads(os.cpu_count() or 4)
m = torch.hub.load('facebookresearch/dinov2', MODEL).eval().to(dev)
if dev == 'cuda': m = m.half()
D = m.embed_dim; log('device', dev, MODEL, D)
# ---- S3 (keys never printed)
cred = {}
for c in sorted(glob.glob('/kaggle/input/**/cred.json', recursive=True)):
    try:
        j = json.load(open(c)); cred = dict(id=j.get('id') or j.get('aws_access_key_id'), key=j.get('key') or j.get('aws_secret_access_key'))
        if cred['id'] and cred['key']: break
    except Exception as e: log('cred.json unreadable', type(e).__name__)
BUCKET = 'train-expds-3'; EP = 's3.ru-7.storage.selcloud.ru'
if cred.get('id'):
    os.environ.update(AWS_ACCESS_KEY_ID=cred['id'], AWS_SECRET_ACCESS_KEY=cred['key'], AWS_S3_ENDPOINT=EP, AWS_VIRTUAL_HOSTING='FALSE',
                      AWS_REGION='ru-7', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif,.tiff', VSI_CACHE='TRUE')
KEYS = {}
try:
    import boto3
    s3 = boto3.client('s3', endpoint_url='https://' + EP, aws_access_key_id=cred.get('id'), aws_secret_access_key=cred.get('key'), region_name='ru-7')
    for pg in s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET):
        for o in pg.get('Contents', []):
            k = o['Key']
            if k.lower().endswith(('.tif', '.tiff')): KEYS.setdefault(k.split('/')[0], []).append(k)
    log('s3 regions', len(KEYS))
except Exception as e: log('s3 listing failed', type(e).__name__)
LAYER_TAG = {'li': ['_Li_ch.', '_Li_c.', '_Li_'], 'ae': ['_Ae_'], 'sp_or': ['_SpOr_', '_Sp_', '_Or_'], 'or': ['_Or_', '_SpOr_'],
             'etgr': ['_EtGr_'], 'maggr': ['_MagGr_'], 'terfl': ['_TerFl_']}
def tif_for(region, mk):
    L = KEYS.get(region, [])
    for t in LAYER_TAG.get(mk, []):
        h = [k for k in L if t in os.path.basename(k)]
        if h: return '/vsis3/' + BUCKET + '/' + sorted(h)[0]
    return None
RELIEF = {os.path.basename(q)[:-4]: q for q in glob.glob('/kaggle/input/**/cnx_relief/*.npy', recursive=True)}
log('relief regions', len(RELIEF))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
def norm(a):
    a = a.astype(np.float32)
    if a.shape[0] == 1: a = np.repeat(a, 3, 0)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99); a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a - MEAN) / STD
class Relief:
    def __init__(s, region):
        s.a = np.load(RELIEF[region], mmap_mode='r'); s.j = json.load(open(RELIEF[region][:-4] + '.json'))
        s.tr = Transformer.from_crs('EPSG:3857', s.j['crs'], always_xy=True); s.b = [s.j['tags'].index(t) for t in ('lrm_s', 'slope', 'svf_r')]
    def crop(s, x, y, half, k=1.0):
        X, Y = s.tr.transform(x, y); st = s.j['step']; c0 = int((X - half - s.j['left']) / st); r0 = int((s.j['top'] - (Y + half)) / st)
        n = max(2, int(2 * half / st)); H, Wd = s.a.shape[1:]
        out = np.zeros((3, n, n), np.uint8); rr0, cc0 = max(r0, 0), max(c0, 0); rr1, cc1 = min(r0 + n, H), min(c0 + n, Wd)
        if rr1 <= rr0 or cc1 <= cc0: return None
        out[:, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = s.a[s.b, rr0:rr1, cc0:cc1]
        t = torch.from_numpy(out.astype(np.float32))[None]; t = torch.nn.functional.interpolate(t, size=(S, S), mode='bilinear', align_corners=False)
        return norm(t[0].numpy())
class Raster:
    def __init__(s, path): s.ds = rasterio.open(path)
    def crop(s, x, y, half, k=1.0):
        h3 = half * k  # metres -> EPSG:3857 units
        xs, ys = wtransform('EPSG:3857', s.ds.crs, [x - h3, x + h3], [y - h3, y + h3])
        win = from_bounds(min(xs), min(ys), max(xs), max(ys), s.ds.transform); nb = min(s.ds.count, 3)
        a = s.ds.read(indexes=list(range(1, nb + 1)), window=win, out_shape=(nb, S, S), boundless=True, fill_value=0, resampling=Resampling.bilinear)
        return norm(a) if a.any() else None
E = np.zeros((N, 2 * D), np.float16); ok = np.zeros(N, bool); cid = np.array([int(r['cid']) for r in rows]); miss = {}
def save(final=False):
    np.save(f'{OUT}/emb.npy', E); np.save(f'{OUT}/ok.npy', ok); np.save(f'{OUT}/cid.npy', cid)
    json.dump(dict(model=MODEL, n=N, ok=int(ok.sum()), sec=round(time.time() - T0), dev=str(dev), final=final, miss=miss), open(f'{OUT}/meta.json', 'w'), ensure_ascii=False)
groups = {}
for i, r in enumerate(rows): groups.setdefault((r['region'], r['markup_type'].lower()), []).append(i)
last = time.time()
def run(X, ii):
    xb = torch.from_numpy(np.stack(X)).to(dev); xb = xb.half() if dev == 'cuda' else xb
    with torch.no_grad():
        f = m.forward_features(xb); e = torch.cat([f['x_norm_clstoken'], f['x_norm_patchtokens'].mean(1)], 1)
    E[ii] = e.float().cpu().numpy().astype(np.float16); ok[ii] = True
for (region, mk), g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if time.time() > DEADLINE: log('deadline'); break
    try:
        if mk == 'li' and region in RELIEF: R = Relief(region)
        else:
            pth = tif_for(region, mk)
            if pth is None: miss[f'{region}|{mk}'] = len(g); continue
            R = Raster(pth)
    except Exception as e: miss[f'{region}|{mk}'] = f'{type(e).__name__}'; continue
    X, ii = [], []
    for i in g:
        r = rows[i]
        # cx,cy are EPSG:3857; w,h are metres in UTM -> widen by the 3857 scale factor
        k = 1 / max(np.cos(np.arctan(np.sinh(float(r['cy']) / 6378137.0))), 0.2)
        half = max(16.0, 1.25 * max(float(r['w']), float(r['h'])))
        try: c = R.crop(float(r['cx']), float(r['cy']), half, k)
        except Exception: c = None
        if c is None: continue
        X.append(c); ii.append(i)
        if len(X) == BS: run(X, ii); X, ii = [], []
    if X: run(X, ii)
    log(region, mk, len(g), 'ok', int(ok.sum()), '/', N)
    if time.time() - last > 1200: save(); last = time.time()
save(True); log('DONE', int(ok.sum()), '/', N)
