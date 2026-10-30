"""koz3-w5-cloudfeat (CI2-1, own code; idea only, no external code).
Point-cloud structure features around every rerank-pool candidate on LAS sites:
ground density, all-point density, share of non-first returns, ground z std, share of empty 0.5 m cells.
Input dataset: cands_cf.csv (cid,region,cx,cy[EPSG:3857],w,h), las_keys.json, laspy wheel, cred.json (never printed).
LAS + UTM.json streamed from S3 one site at a time (prefetch 1). Output: cf_<site>.csv per site, cloudfeat.csv, log.json."""
import os, sys, json, glob, time, csv, threading, subprocess, math
import numpy as np
T0 = time.time(); DEADLINE_H = float(os.environ.get('DEADLINE_H', 11.0))
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
IN = os.environ.get('CF_IN') or os.path.dirname(glob.glob('/kaggle/input/**/cands_cf.csv', recursive=True)[0])
OUT = os.environ.get('CF_OUT', '/kaggle/working'); os.makedirs(OUT, exist_ok=True)
TMP = os.environ.get('CF_TMP', '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp/cf'); os.makedirs(TMP, exist_ok=True)
try: import laspy
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', *glob.glob(IN + '/laspy-*.whl')], check=True); import laspy
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from pyproj import Transformer
# ---- S3 (keys from cred.json or env; never printed)
cid_, key_ = os.environ.get('AWS_ACCESS_KEY_ID'), os.environ.get('AWS_SECRET_ACCESS_KEY')
for c in sorted(glob.glob(IN + '/cred.json')):
    j = json.load(open(c)); cid_ = j.get('id') or j.get('aws_access_key_id') or cid_; key_ = j.get('key') or j.get('aws_secret_access_key') or key_
BUCKET = os.environ.get('S3_BUCKET', 'train-expds-3'); EP = os.environ.get('S3_ENDPOINT', 'https://s3.ru-7.storage.selcloud.ru')
if not EP.startswith('http'): EP = 'https://' + EP
s3 = boto3.client('s3', endpoint_url=EP, aws_access_key_id=cid_, aws_secret_access_key=key_, region_name='ru-7',
                  config=Config(signature_version='s3v4', retries={'max_attempts': 10}))
try: s3.head_bucket(Bucket=BUCKET); log('s3 ok')
except Exception as e: log('S3 UNREACHABLE (internet off?)', type(e).__name__); sys.exit(3)
TC = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=8)
# ---- inputs
KEYS = json.load(open(IN + '/las_keys.json', encoding='utf-8'))
C = {}
with open(IN + '/cands_cf.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f): C.setdefault(r['region'], []).append((int(r['cid']), float(r['cx']), float(r['cy']), float(r['w']), float(r['h'])))
ONLY = os.environ.get('CF_ONLY'); sites = sorted(C, key=lambda s: KEYS[s]['size'])
if ONLY: sites = [s for s in sites if s in ONLY.split(',')]
log(len(sites), 'sites', sum(len(C[s]) for s in sites), 'cands')
R = 0.5; RING = 8.0; RMIN, RMAX = 2.0, 15.0
NAMES = ['cf_gd_in', 'cf_gd_ctx', 'cf_pd_in', 'cf_nf_in', 'cf_nf_ctx', 'cf_zs_in', 'cf_zs_ctx', 'cf_hole_in', 'cf_hole_ctx', 'cf_gd_ratio']
def fetch(site):
    d = f'{TMP}/{site.split("_")[0]}'; os.makedirs(d, exist_ok=True)
    s3.download_file(BUCKET, KEYS[site]['utm'], d + '/UTM.json'); s3.download_file(BUCKET, KEYS[site]['las'], d + '/p.las', Config=TC)
    return d
OFF = {}
def offsets(rin):
    if rin not in OFF:
        n = int(math.ceil((rin + RING) / R)); dy, dx = np.mgrid[-n:n + 1, -n:n + 1]; d = R * np.hypot(dx, dy).ravel()
        k = d <= rin + RING; OFF[rin] = (dx.ravel()[k], dy.ravel()[k], d[k] <= rin)
    return OFF[rin]
def process(site, d):
    t = time.time(); crs = json.load(open(d + '/UTM.json'))['crs'].split(':')[-1]
    cand = np.array(C[site]); ids = cand[:, 0].astype(np.int64)
    ux, uy = Transformer.from_crs(3857, int(crs), always_xy=True).transform(cand[:, 1], cand[:, 2])
    # sizes w,h are metres in UTM; radius of the disc
    rin = np.clip(np.round(0.5 * np.maximum(cand[:, 3], cand[:, 4]) / R) * R, RMIN, RMAX)
    fh = laspy.open(d + '/p.las'); h = fh.header; x0, y0 = math.floor(h.mins[0]), math.floor(h.mins[1])
    W = int((h.maxs[0] - x0) / R) + 2; H = int((h.maxs[1] - y0) / R) + 2
    c0 = np.floor((ux - x0) / R).astype(np.int64); r0 = np.floor((uy - y0) / R).astype(np.int64)
    # ROI = union of all candidate discs+rings (linear cell ids)
    U = np.zeros(0, np.int64)
    for rv in np.unique(rin):
        dx, dy, _ = offsets(float(rv)); jj = np.flatnonzero(rin == rv)
        for b in range(0, len(jj), max(1, 2_000_000 // len(dx))):
            q = jj[b:b + max(1, 2_000_000 // len(dx))]
            cc = c0[q, None] + dx[None]; rr = r0[q, None] + dy[None]; ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
            U = np.union1d(U, (rr * W + cc)[ok])
    n = len(U); na = np.zeros(n); ng = np.zeros(n); nf = np.zeros(n); sz = np.zeros(n); sz2 = np.zeros(n); zr = float(h.mins[2])
    npts = 0
    for p in fh.chunk_iterator(5_000_000):
        x = np.asarray(p.x); y = np.asarray(p.y); cl = np.asarray(p.classification); npts += len(x)
        col = ((x - x0) / R).astype(np.int64); row = ((y - y0) / R).astype(np.int64)
        lin = row * W + col; i = np.searchsorted(U, lin); i[i >= n] = n - 1 if n else 0
        m = (U[i] == lin) & (cl != 7) & (cl != 18) if n else np.zeros(len(x), bool)
        if not m.any(): continue
        i = i[m]; cl = cl[m]; z = np.asarray(p.z)[m] - zr; rn = np.asarray(p.return_number)[m]
        g = (cl == 2) | (cl == 8)
        na += np.bincount(i, minlength=n); nf += np.bincount(i, weights=(rn > 1).astype(float), minlength=n)
        ng += np.bincount(i[g], minlength=n); sz += np.bincount(i[g], weights=z[g], minlength=n); sz2 += np.bincount(i[g], weights=z[g] ** 2, minlength=n)
    fh.close()
    F = np.full((len(ids), len(NAMES)), np.nan, np.float32)
    def agg(sel, cnt):  # sel: (k, m) idx into U or -1, cnt mask
        v = lambda a: np.where(sel >= 0, a[np.maximum(sel, 0)], 0.0) * cnt
        A, G, NF, S, S2 = v(na).sum(1), v(ng).sum(1), v(nf).sum(1), v(sz).sum(1), v(sz2).sum(1)
        cells = cnt.sum(1); hole = (((v(ng) == 0) & cnt).sum(1)) / np.maximum(cells, 1)
        var = np.where(G > 2, S2 / np.maximum(G, 1) - (S / np.maximum(G, 1)) ** 2, np.nan)
        return G / (cells * R * R), A / (cells * R * R), np.where(A > 0, NF / np.maximum(A, 1), np.nan), np.sqrt(np.maximum(var, 0)), hole
    for rv in np.unique(rin):
        dx, dy, inn = offsets(float(rv)); jj = np.flatnonzero(rin == rv); bs = max(1, 3_000_000 // len(dx))
        for b in range(0, len(jj), bs):
            q = jj[b:b + bs]; cc = c0[q, None] + dx[None]; rr = r0[q, None] + dy[None]; ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
            lin = rr * W + cc; s = np.searchsorted(U, lin); s[s >= n] = 0; s = np.where(ok & (U[s] == lin) if n else False, s, -1)
            gi, pi, nfi, zi, hi = agg(s, np.broadcast_to(inn[None], s.shape)); go, po, nfo, zo, ho = agg(s, np.broadcast_to(~inn[None], s.shape))
            F[q] = np.stack([gi, go, pi, nfi, nfo, zi, zo, hi, ho, gi / np.maximum(go, 1e-3)], 1)
    with open(f'{OUT}/cf_{site.split("_")[0]}.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['cid'] + NAMES)
        for c, r in zip(ids, F): w.writerow([int(c)] + [('' if not np.isfinite(v) else f'{v:.5g}') for v in r])
    return dict(site=site, cands=len(ids), pts=int(npts), roi_cells=int(n), sec=round(time.time() - t, 1))
LOG = []; nxt = {}
def pre(s):
    try: nxt[s] = fetch(s)
    except Exception as e: nxt[s] = e
th = None
for k, s in enumerate(sites):
    if os.path.exists(f'{OUT}/cf_{s.split("_")[0]}.csv'): continue
    if time.time() - T0 > DEADLINE_H * 3600: log('deadline'); break
    if th is None: pre(s)
    else: th.join()
    th = threading.Thread(target=pre, args=(sites[k + 1],)) if k + 1 < len(sites) else None
    if th: th.start()
    d = nxt.pop(s)
    if isinstance(d, Exception): log(s, 'download failed', type(d).__name__); LOG.append(dict(site=s, err='download ' + type(d).__name__)); continue
    try: st = process(s, d); LOG.append(st); log(st)
    except Exception as e: log(s, 'FAILED', type(e).__name__, str(e)[:200]); LOG.append(dict(site=s, err=type(e).__name__ + ' ' + str(e)[:200]))
    subprocess.run(['rm', '-rf', d])
    json.dump(LOG, open(f'{OUT}/log.json', 'w'), ensure_ascii=False, indent=0)
if th: th.join()
rows = 0
with open(f'{OUT}/cloudfeat.csv', 'w', newline='') as g:
    w = None
    for p in sorted(glob.glob(f'{OUT}/cf_*.csv')):
        with open(p) as f:
            r = csv.reader(f); hdr = next(r)
            if w is None: w = csv.writer(g); w.writerow(hdr)
            for x in r: w.writerow(x); rows += 1
json.dump(LOG, open(f'{OUT}/log.json', 'w'), ensure_ascii=False, indent=0)
log('DONE rows', rows, 'sites ok', sum(1 for x in LOG if 'err' not in x), 'failed', [x['site'] for x in LOG if 'err' in x])
