# bot-5 Ae barrow candidates + hand-crafted features (NO pretrained weights).
# One pass per Ae tif: WarpedVRT -> region UTM at 1.0 m, tiles 2048 px (+64 margin),
# local contrast normalisation (sigma 25 m). Candidates per tile:
#   src=0 own LoG peaks (|sigma^2 LoG| max over sigma 2..6.5 m, NMS 7 px, R>0.05, top D_MAX/km2 of valid area)
#   src=1 existing stack rows (table.pkl, kurgan classes, mk=='ae') inside the tile core
#   src=2 external candidates (env CANDS=json list with reg, ux, uy[, cand_id]) e.g. from bot-7
# Features sampled at the candidate pixel from 1 m maps and 97x97 px (96 m) patches.
# DINO windows (64x64 px = 64 m, uint8 of normalised image) saved for src 1/2, for gen peaks inside
# any GT barrow (only to make AUC measurable for variant 2; labels are NOT features) and a random 8% of gen peaks.
import os, sys, json, glob, time, pickle, numpy as np, rasterio, shapely
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform
from scipy import ndimage as ndi
import cv2
cv2.setNumThreads(2)
DATA = '/workspace/data/train-expds-3'
OUT = os.environ.get('OUT', '/workspace/koz3_shared/bot-5/tmp/kurg_ae/feat')
EXT_ONLY = os.environ.get('EXT_ONLY') == '1'; WIN_FRAC = float(os.environ.get('WIN_FRAC', 0.08))
RES = 1.0; CORE = 2048; MARG = 64; D_MAX = float(os.environ.get('D_MAX', 1000))
SIGMAS = (2.0, 3.0, 4.5, 6.5, 9.0); GEN_S = 4
CLS = ('kurgany_povrezhdennye', 'kurgany_tselye')
P = 48  # patch half size -> 97x97
yy_, xx_ = np.mgrid[-P:P + 1, -P:P + 1].astype(np.float32)
rr_ = np.hypot(xx_, yy_); th_ = np.arctan2(yy_, xx_)
RINGS = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 13), (13, 18), (18, 25), (25, 35), (35, 48)]
EN_RINGS = [(0, 6), (6, 13), (13, 25), (25, 48)]
def ringk(a, b):
    k = ((rr_ >= a) & (rr_ < b)).astype(np.float32) if a > 0 else (rr_ < b).astype(np.float32)
    return k
KR = np.stack([ringk(a, b) for a, b in RINGS]).reshape(len(RINGS), -1)
KE = np.stack([ringk(a, b) for a, b in EN_RINGS]).reshape(len(EN_RINGS), -1)
HARM = []
for a, b in [(2, 8), (8, 16)]:
    k = ringk(a, b)
    for h in (1, 2):
        HARM.append(k * np.cos(h * th_)); HARM.append(k * np.sin(h * th_))
KH = np.stack(HARM).reshape(len(HARM), -1)
FLUX = []
for a, b in [(2, 12), (4, 20)]:
    k = ringk(a, b) / np.maximum(rr_, 1e-3)
    FLUX.append((k * xx_, k * yy_))
BOX = {s: (np.abs(xx_) <= s / 2) & (np.abs(yy_) <= s / 2) for s in (16, 32, 64)}

def read_gray(vrt, win, src_info):
    cnt, dtype, nod = src_info
    bands = [1, 2, 3] if cnt >= 3 else [1]
    c0, r0, w, h = int(win.col_off), int(win.row_off), int(win.width), int(win.height)
    a0, b0 = max(c0, 0), max(r0, 0); a1, b1 = min(c0 + w, vrt.width), min(r0 + h, vrt.height)
    g = np.zeros((h, w), np.float32); m = np.zeros((h, w), bool)
    if a1 <= a0 or b1 <= b0:
        return g, m
    ww = Window(a0, b0, a1 - a0, b1 - b0)
    a = vrt.read(bands, window=ww).astype(np.float32)
    mm = vrt.read_masks(1, window=ww) > 0
    if cnt == 4:
        mm &= vrt.read(4, window=ww) > 0
    if nod is not None:
        for b in range(a.shape[0]):
            mm &= a[b] != nod
    if dtype == 'int16':
        mm &= a[0] > -32000
    g[b0 - r0:b1 - r0, a0 - c0:a1 - c0] = a.mean(0)
    m[b0 - r0:b1 - r0, a0 - c0:a1 - c0] = mm
    return g, m

def normalise(g, m, s=25.0):
    v = m.astype(np.float32)
    gv = ndi.gaussian_filter(v, s) + 1e-3
    mu = ndi.gaussian_filter(g * v, s) / gv
    var = ndi.gaussian_filter(((g - mu) ** 2) * v, s) / gv
    n = (g - mu) / (np.sqrt(var) + 1e-3)
    n = np.clip(n, -5, 5); n[~m] = 0
    return n.astype(np.float32)

def frst(gx, gy, mag, r):
    H, W = mag.shape
    thr = np.percentile(mag, 70)
    yy, xx = np.nonzero(mag > thr)
    ux = gx[yy, xx] / (mag[yy, xx] + 1e-6); uy = gy[yy, xx] / (mag[yy, xx] + 1e-6)
    O = np.zeros(H * W, np.float32); Mm = np.zeros(H * W, np.float32)
    for sgn in (1, -1):
        px = np.clip(np.round(xx + sgn * r * ux).astype(int), 0, W - 1)
        py = np.clip(np.round(yy + sgn * r * uy).astype(int), 0, H - 1)
        idx = py * W + px
        O += sgn * np.bincount(idx, minlength=H * W).astype(np.float32)
        Mm += sgn * np.bincount(idx, weights=mag[yy, xx], minlength=H * W).astype(np.float32)
    kn = 9.9
    O = np.clip(O, -kn, kn).reshape(H, W); Mm = Mm.reshape(H, W)
    F = (Mm / kn) * (np.abs(O) / kn) ** 2
    return ndi.gaussian_filter(F, 0.25 * r)

def lbp(img):
    c = img[1:-1, 1:-1]
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    bits = [(img[1 + dy:img.shape[0] - 1 + dy, 1 + dx:img.shape[1] - 1 + dx] >= c) for dy, dx in offs]
    B = np.stack(bits).astype(np.int8)
    U = np.abs(np.diff(np.concatenate([B, B[:1]]), axis=0)).sum(0)
    code = np.where(U <= 2, B.sum(0), 9).astype(np.int8)
    out = np.full(img.shape, 9, np.int8); out[1:-1, 1:-1] = code
    return out

FEAT_NAMES = (['gen_R', 'gen_scale', 'gen_sign', 'valid32', 'gsd']
    + [f'log{s}' for s in SIGMAS] + [f'logmax{s}' for s in SIGMAS] + ['dog1', 'dog2']
    + [f'frst{r}' for r in (3, 5, 8)] + [f'frstmax{r}' for r in (3, 5, 8)]
    + [f'ring{a}_{b}' for a, b in RINGS] + [f'ringsd{a}_{b}' for a, b in EN_RINGS]
    + ['c04_613', 'c06_1325'] + ['h1_2_8', 'h2_2_8', 'h1_8_16', 'h2_8_16']
    + ['flux2_12', 'flux4_20'] + [f'gmag{s}' for s in (16, 32, 64)]
    + [f'lbp{i}' for i in range(10)]
    + ['glcm_con_x', 'glcm_con_y', 'glcm_hom_x', 'glcm_hom_y', 'glcm_cor_x', 'glcm_cor_y', 'glcm_con64'])

def gather(M, cy, cx):
    iy = cy[:, None, None] + np.arange(-P, P + 1)[None, :, None]
    ix = cx[:, None, None] + np.arange(-P, P + 1)[None, None, :]
    return M[iy, ix].reshape(len(cy), -1)

def tile_features(n, m, cy, cx, extra):
    # maps
    L = np.stack([s * s * ndi.gaussian_laplace(n, s) for s in SIGMAS])
    Lmax = np.stack([ndi.maximum_filter(np.abs(l), 7) for l in L])
    dog1 = ndi.gaussian_filter(n, 1.5) - ndi.gaussian_filter(n, 4.0)
    dog2 = ndi.gaussian_filter(n, 4.0) - ndi.gaussian_filter(n, 10.0)
    gs = ndi.gaussian_filter(n, 1.0)
    gx = ndi.sobel(gs, 1); gy = ndi.sobel(gs, 0); mag = np.hypot(gx, gy)
    FR = [frst(gx, gy, mag, r) for r in (3, 5, 8)]
    FRm = [ndi.maximum_filter(np.abs(f), 7) for f in FR]
    code = lbp(ndi.gaussian_filter(n, 0.7))
    q = np.clip(np.floor((n + 2.5) / 5 * 8), 0, 7).astype(np.float32)
    N = len(cy); F = np.zeros((N, len(FEAT_NAMES)), np.float32)
    F[:, :5] = extra
    F[:, 3] = 0
    j = 5
    for l in L: F[:, j] = l[cy, cx]; j += 1
    for l in Lmax: F[:, j] = l[cy, cx]; j += 1
    F[:, j] = dog1[cy, cx]; F[:, j + 1] = dog2[cy, cx]; j += 2
    for f in FR: F[:, j] = f[cy, cx]; j += 1
    for f in FRm: F[:, j] = f[cy, cx]; j += 1
    qx = (q - np.roll(q, -1, 1)); qy = (q - np.roll(q, -1, 0))
    qq_x = q * np.roll(q, -1, 1); qq_y = q * np.roll(q, -1, 0)
    mv = m.astype(np.float32)
    b32 = BOX[32].reshape(-1); b16 = BOX[16].reshape(-1); b64 = BOX[64].reshape(-1)
    for s0 in range(0, N, 600):
        sl = slice(s0, min(N, s0 + 600)); a, b = cy[sl], cx[sl]
        pn = gather(n, a, b); k = j
        ring = pn @ KR.T / KR.sum(1); F[sl, k:k + 9] = ring; k += 9
        en = (pn ** 2) @ KE.T / KE.sum(1)
        mu = pn @ KE.T / KE.sum(1); sd = np.sqrt(np.maximum(en - mu ** 2, 0)); F[sl, k:k + 4] = sd; k += 4
        F[sl, k] = ring[:, :2].mean(1) - ring[:, 3:5].mean(1)
        F[sl, k + 1] = ring[:, :3].mean(1) - ring[:, 5:7].mean(1); k += 2
        H = pn @ KH.T
        for t, sdi in ((0, 0), (4, 1)):
            for h in range(2):
                F[sl, k] = np.hypot(H[:, t + 2 * h], H[:, t + 2 * h + 1]) / (KH[t].__abs__().sum() * (sd[:, sdi] + 1e-3)); k += 1
        pgx = gather(gx, a, b); pgy = gather(gy, a, b)
        for kx, ky in FLUX:
            F[sl, k] = (pgx @ kx.reshape(-1) + pgy @ ky.reshape(-1)) / kx.size; k += 1
        pm = gather(mag, a, b)
        for bb in (b16, b32, b64): F[sl, k] = pm[:, bb].mean(1); k += 1
        pc = gather(code, a, b)[:, b32].astype(np.int64)
        oh = np.zeros((len(a), 10), np.float32)
        np.add.at(oh, (np.repeat(np.arange(len(a)), pc.shape[1]), pc.reshape(-1)), 1)
        F[sl, k:k + 10] = oh / pc.shape[1]; k += 10
        px_, py_ = gather(qx, a, b)[:, b32], gather(qy, a, b)[:, b32]
        F[sl, k] = (px_ ** 2).mean(1); F[sl, k + 1] = (py_ ** 2).mean(1)
        F[sl, k + 2] = (1 / (1 + px_ ** 2)).mean(1); F[sl, k + 3] = (1 / (1 + py_ ** 2)).mean(1)
        pq = gather(q, a, b)[:, b32]; mq = pq.mean(1); vq = pq.var(1) + 1e-3
        F[sl, k + 4] = (gather(qq_x, a, b)[:, b32].mean(1) - mq ** 2) / vq
        F[sl, k + 5] = (gather(qq_y, a, b)[:, b32].mean(1) - mq ** 2) / vq
        F[sl, k + 6] = (gather(qx, a, b)[:, b64] ** 2).mean(1)
        F[sl, 3] = gather(mv, a, b)[:, b32].mean(1)
    return F

def file_gsd(src):
    gsd = abs(src.transform.a)
    if src.crs.to_epsg() == 3857:
        import pyproj
        lat = pyproj.Transformer.from_crs(3857, 4326, always_xy=True).transform(*src.xy(src.height // 2, src.width // 2))[1]
        gsd *= np.cos(np.radians(lat))
    return gsd

def run_file(f, stack_xy, ext_xy, gt_geoms, rng):
    regdir = f.split('/')[4]
    utm = json.load(open(glob.glob(f'{DATA}/{regdir}/*UTM*.json')[0]))['crs'].split('::')[-1]
    recs, feats, wins, winidx = [], [], [], []
    with rasterio.open(f) as src:
        info = (src.count, src.dtypes[0], src.nodata); gsd = file_gsd(src)
        tr, w_, h_ = calculate_default_transform(src.crs, f'EPSG:{utm}', src.width, src.height, *src.bounds, resolution=(RES, RES))
        with WarpedVRT(src, crs=f'EPSG:{utm}', transform=tr, width=w_, height=h_, resampling=Resampling.average) as vrt:
            H, W = vrt.height, vrt.width; T = vrt.transform
            for y0 in range(0, H, CORE):
                for x0 in range(0, W, CORE):
                    win = Window(x0 - MARG, y0 - MARG, CORE + 2 * MARG, CORE + 2 * MARG)
                    g, m = read_gray(vrt, win, info)
                    if m[MARG:-MARG, MARG:-MARG].mean() < 0.002:
                        continue
                    n = normalise(g, m)
                    Lg = np.stack([s * s * ndi.gaussian_laplace(n, s) for s in SIGMAS[:GEN_S]])
                    A = np.abs(Lg); k = A.argmax(0); R = np.take_along_axis(A, k[None], 0)[0]
                    sg = np.sign(np.take_along_axis(Lg, k[None], 0)[0]); del Lg, A
                    ok = ndi.binary_erosion(m, iterations=8)
                    pk = (R == ndi.maximum_filter(R, 7)) & ok & (R > 0.05)
                    pk[:MARG] = pk[-MARG:] = False; pk[:, :MARG] = pk[:, -MARG:] = False
                    yy, xx = np.nonzero(pk)
                    area = ok[MARG:-MARG, MARG:-MARG].sum() * RES * RES / 1e6
                    K = int(np.ceil(D_MAX * area))
                    if len(yy) > K:
                        o = np.argsort(-R[yy, xx])[:K]; yy, xx = yy[o], xx[o]
                    if EXT_ONLY: yy, xx = yy[:0], xx[:0]
                    rows = [(yy, xx, np.zeros(len(yy), int), np.full(len(yy), -1))]
                    # stack / external candidates in the core of this tile
                    for srcid, xy in ((1, stack_xy), (2, ext_xy)):
                        if xy is None or not len(xy): continue
                        cxp = (xy[:, 0] - T.c) / T.a - (x0 - MARG); cyp = (xy[:, 1] - T.f) / T.e - (y0 - MARG)
                        cxi = np.floor(cxp).astype(int); cyi = np.floor(cyp).astype(int)
                        sel = (cxi >= MARG) & (cxi < MARG + CORE) & (cyi >= MARG) & (cyi < MARG + CORE)
                        sel &= (cxi < g.shape[1] - MARG) & (cyi < g.shape[0] - MARG)
                        sel_i = np.flatnonzero(sel)
                        sel_i = sel_i[m[cyi[sel_i], cxi[sel_i]]]
                        rows.append((cyi[sel_i], cxi[sel_i], np.full(len(sel_i), srcid), xy[sel_i, 2].astype(int)))
                    cy = np.concatenate([r[0] for r in rows]); cx = np.concatenate([r[1] for r in rows])
                    if not len(cy): continue
                    srcs = np.concatenate([r[2] for r in rows]); ids = np.concatenate([r[3] for r in rows])
                    extra = np.stack([R[cy, cx], k[cy, cx], sg[cy, cx], np.zeros(len(cy)), np.full(len(cy), gsd)], 1)
                    F = tile_features(n, m, cy, cx, extra)
                    ux = T.c + (x0 - MARG + cx + 0.5) * T.a; uy = T.f + (y0 - MARG + cy + 0.5) * T.e
                    # DINO windows
                    keepw = srcs == 1 if EXT_ONLY else srcs > 0
                    if gt_geoms is not None and len(gt_geoms):
                        inside = shapely.STRtree(gt_geoms).query(shapely.points(ux, uy), predicate='within')[0]
                        keepw[inside] = True
                    keepw |= rng.random(len(cy)) < WIN_FRAC
                    wi = np.flatnonzero(keepw)
                    if len(wi):
                        pw = gather(n, cy[wi], cx[wi]).reshape(len(wi), 2 * P + 1, 2 * P + 1)[:, 16:80, 16:80]
                        wins.append(np.clip(pw * 40 + 128, 0, 255).astype(np.uint8))
                    winidx.append(keepw)
                    recs.append(np.stack([ux, uy, srcs, ids, np.full(len(cy), area)], 1)); feats.append(F)
    if not recs:
        return regdir, utm, gsd, np.zeros((0, 5)), np.zeros((0, len(FEAT_NAMES)), np.float32), np.zeros((0, 64, 64), np.uint8), np.zeros(0, bool)
    return (regdir, utm, gsd, np.concatenate(recs), np.concatenate(feats),
            np.concatenate(wins) if wins else np.zeros((0, 64, 64), np.uint8), np.concatenate(winidx))

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f'{DATA}/*/*Ae*/*.tif'))
    lim = os.environ.get('LIMIT_FILES')
    if lim: files = [f for f in files if any(r in f for r in lim.split(','))]
    Tb = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
    cidx = [Tb['CL'].index(c) for c in CLS]
    srow = np.flatnonzero(np.isin(Tb['cls'], cidx) & (Tb['mk'] == 'ae'))
    sreg = Tb['reg'][srow]; sx = Tb['cx'][srow]; sy = Tb['cy'][srow]
    Wk = pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))
    gtg = {}
    for C in CLS:
        g = shapely.from_wkb(Wk[C]); gr = np.array(Tb['greg'][C])
        for r in set(gr.tolist()): gtg.setdefault(r, []).extend(g[gr == r].tolist())
    del Tb
    ext = None
    if os.environ.get('CANDS'):
        if os.environ['CANDS'].endswith('.npz'):
            Z = np.load(os.environ['CANDS'])
            ext = dict(reg=Z['reg'], tif=Z['tif'], ux=Z['ux'], uy=Z['uy'], id=Z['cand_id'])
        else:
            E = json.load(open(os.environ['CANDS']))
            ext = dict(reg=np.array([c['reg'] for c in E]), tif=np.array([c.get('file', '') for c in E]),
                       ux=np.array([c['ux'] for c in E]), uy=np.array([c['uy'] for c in E]),
                       id=np.array([c.get('cand_id', i) for i, c in enumerate(E)])); del E
        if EXT_ONLY:
            files = [f for f in files if (ext['reg'] == f.split('/')[4]).any()]
    rng = np.random.default_rng(0)
    json.dump(FEAT_NAMES, open(f'{OUT}/feat_names.json', 'w'))
    for f in files:
        t = time.time()
        reg = f.split('/')[4]
        key = reg + '__' + os.path.basename(f)[:-4]
        out = f'{OUT}/{key}.npz'
        if os.path.exists(out): continue
        mk = sreg == reg
        stack_xy = np.stack([sx[mk], sy[mk], srow[mk]], 1)
        ext_xy = None
        if ext is not None:
            me = (ext['reg'] == reg) & ((ext['tif'] == os.path.basename(f)) | (ext['tif'] == ''))
            ext_xy = np.stack([ext['ux'][me], ext['uy'][me], ext['id'][me].astype(float)], 1) if me.any() else None
        r_, utm, gsd, rec, F, wins, widx = run_file(f, stack_xy, ext_xy, np.array(gtg.get(reg, []), dtype=object) if reg in gtg else None, rng)
        np.savez(out, rec=rec, F=F.astype(np.float16), wins=wins, widx=widx, reg=reg, tif=f, utm=utm, gsd=gsd)
        print(f'{key} n={len(rec)} stack={int((rec[:,2]==1).sum()) if len(rec) else 0} ext={int((rec[:,2]==2).sum()) if len(rec) else 0} wins={len(wins)} gsd={gsd:.3f} t={time.time()-t:.0f}s', flush=True)
    print('DONE')
