"""Окна DEM (0,25 м, среднее z земли) из LAZ вокруг кандидатов gtcand*.npz. xfer_venv (laspy+rasterio).
Запуск: python dem_windows.py ../val/gtcand.npz win.npz"""
import sys, glob, json, gc, numpy as np, laspy
from rasterio.warp import transform
D = np.load(sys.argv[1], allow_pickle=True)
reg, cx, cy = D['reg'], D['cx'], D['cy']
PX, H = 0.25, 20.0; n = int(2 * H / PX)
W = np.full((len(reg), n, n), np.nan, np.float32); ok = np.zeros(len(reg), bool)
for rg in sorted(set(reg.tolist())):
    ls = glob.glob(f'/workspace/data/train-expds-3/{rg}/01_*_La_точки/*.laz')
    if not ls: continue
    crs = 'EPSG:' + json.load(open(glob.glob(f'/workspace/data/train-expds-3/{rg}/UTM.json')[0]))['crs'].split(':')[-1]
    sel = np.flatnonzero(reg == rg)
    X, Y = transform('EPSG:3857', crs, cx[sel].tolist(), cy[sel].tolist())
    X, Y = np.array(X), np.array(Y)
    with laspy.open(ls[0]) as fh:
        for ch in fh.chunk_iterator(2_000_000):
            px, py, pz = np.asarray(ch.x), np.asarray(ch.y), np.asarray(ch.z, np.float32)
            if 'classification' in ch.point_format.dimension_names:
                g = np.asarray(ch.classification) == 2
                if g.any(): px, py, pz = px[g], py[g], pz[g]
            for k, p in enumerate(sel):
                m = (np.abs(px - X[k]) < H) & (np.abs(py - Y[k]) < H)
                if not m.any(): continue
                ix = ((px[m] - (X[k] - H)) / PX).astype(int); iy = ((Y[k] + H - py[m]) / PX).astype(int)
                ii = np.clip(iy, 0, n - 1) * n + np.clip(ix, 0, n - 1)
                s = np.bincount(ii, pz[m], n * n); c = np.bincount(ii, None, n * n)
                prev = W[p].ravel(); acc = np.where(np.isnan(prev), 0, prev)
                # накопление сумм: храним сумму в W, счёт в C
                if 'C' not in globals(): globals()['C'] = np.zeros((len(reg), n * n), np.float32)
                W[p] = (acc + s).reshape(n, n); C[p] += c; ok[p] = True
            del px, py, pz; gc.collect()
    print(rg, len(sel), int(ok[sel].sum()), flush=True)
C = globals().get('C', np.zeros((len(reg), n * n), np.float32))
Z = np.where(C.reshape(W.shape) > 0, W / np.maximum(C.reshape(W.shape), 1), np.nan).astype(np.float32)
np.savez_compressed(sys.argv[2], Z=Z, ok=ok, PX=PX, H=H, **{k: D[k] for k in D.files})
print('ok', int(ok.sum()), 'из', len(reg))
