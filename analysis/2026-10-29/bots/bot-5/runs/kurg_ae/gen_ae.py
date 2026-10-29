# Pass 1: simple LoG blob generator on Ae rasters (debug candidates for bot-5 classifier).
# Reads each Ae tif via WarpedVRT into region UTM at 1.0 m, tiles 2048 px (+64 margin),
# local contrast normalisation (sigma 25 m), scale-normalised |LoG| at sigma 2/3/4.5/6.5 m,
# NMS 7 px, keeps top D_MAX per km2 of valid area per tile. No pretrained weights, no GT used.
import os, sys, json, glob, time, numpy as np, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage as ndi
DATA = '/workspace/data/train-expds-3'
OUT = '/workspace/koz3_shared/bot-5/tmp/kurg_ae/gen'
RES = 1.0; CORE = 2048; MARG = 64; D_MAX = float(os.environ.get("D_MAX", 3000))
SIGMAS = (2.0, 3.0, 4.5, 6.5)

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
    return n

def run_file(f):
    regdir = f.split('/')[-3] if False else f.split('/')[len(DATA.split('/'))]
    utm = json.load(open(glob.glob(f'{DATA}/{regdir}/*UTM*.json')[0]))['crs'].split('::')[-1]
    with rasterio.open(f) as src:
        info = (src.count, src.dtypes[0], src.nodata)
        from rasterio.warp import calculate_default_transform
        tr, w_, h_ = calculate_default_transform(src.crs, f'EPSG:{utm}', src.width, src.height, *src.bounds, resolution=(RES, RES))
        with WarpedVRT(src, crs=f'EPSG:{utm}', transform=tr, width=w_, height=h_, resampling=Resampling.average) as vrt:
            assert abs(vrt.transform.a - RES) < 1e-6, vrt.transform
            H, W = vrt.height, vrt.width; T = vrt.transform
            rows = []
            for y0 in range(0, H, CORE):
                for x0 in range(0, W, CORE):
                    win = Window(x0 - MARG, y0 - MARG, CORE + 2 * MARG, CORE + 2 * MARG)
                    g, m = read_gray(vrt, win, info)
                    if m[MARG:-MARG, MARG:-MARG].mean() < 0.01:
                        continue
                    n = normalise(g, m)
                    L = np.stack([s * s * ndi.gaussian_laplace(n, s) for s in SIGMAS])
                    A = np.abs(L); k = A.argmax(0); R = np.take_along_axis(A, k[None], 0)[0]
                    sg = np.sign(np.take_along_axis(L, k[None], 0)[0])
                    ok = ndi.binary_erosion(m, iterations=8)
                    pk = (R == ndi.maximum_filter(R, 7)) & ok & (R > 0.05)
                    pk[:MARG] = pk[-MARG:] = False; pk[:, :MARG] = pk[:, -MARG:] = False
                    yy, xx = np.nonzero(pk)
                    area = ok[MARG:-MARG, MARG:-MARG].sum() * RES * RES / 1e6
                    K = int(np.ceil(D_MAX * area))
                    if len(yy) > K:
                        o = np.argsort(-R[yy, xx])[:K]; yy, xx = yy[o], xx[o]
                    px = x0 - MARG + xx + 0.5; py = y0 - MARG + yy + 0.5
                    ux = T.c + px * T.a; uy = T.f + py * T.e
                    rows.append(np.stack([ux, uy, R[yy, xx], k[yy, xx], sg[yy, xx], np.full(len(yy), area)], 1))
            arr = np.concatenate(rows) if rows else np.zeros((0, 6))
    gsd = None
    with rasterio.open(f) as src:
        gsd = abs(src.transform.a)
        if src.crs.to_epsg() == 3857:
            import pyproj
            lat = pyproj.Transformer.from_crs(3857, 4326, always_xy=True).transform(*src.xy(src.height // 2, src.width // 2))[1]
            gsd *= np.cos(np.radians(lat))
    return regdir, utm, arr, gsd

if __name__ == '__main__':
    files = sorted(glob.glob(f'{DATA}/*/*Ae*/*.tif'))
    lim = os.environ.get('LIMIT_FILES')
    if lim: files = [f for f in files if any(r in f for r in lim.split(','))]
    for f in files:
        t = time.time()
        key = f.split('/')[-3] + '__' + os.path.basename(f)[:-4]
        out = f'{OUT}/{key}.npz'
        if os.path.exists(out):
            continue
        reg, utm, arr, gsd = run_file(f)
        np.savez(out, cands=arr, reg=reg, tif=f, utm=utm, gsd=gsd)
        print(f'{key} n={len(arr)} gsd={gsd:.3f} t={time.time()-t:.0f}s', flush=True)
    print('DONE')
