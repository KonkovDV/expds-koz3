"""Проверка привязки к рельефу по DEM из LAZ (окна из dem_windows.py). dfine_venv (scipy, rasterio).
Те же варианты, что snap_li.py: PK1/PK2 (argmax LRM в круге 3 м после σ 1/2 м), CM (mean-shift положительного рельефа).
LRM = DEM − гаусс(σ=TR). Ошибка = расстояние до центра эталона по земле (UTM, м)."""
import sys, glob, json, numpy as np
from scipy import ndimage as ndi
from rasterio.warp import transform
D = np.load(sys.argv[1], allow_pickle=True)
Z, ok, PX, H = D['Z'], D['ok'], float(D['PX']), float(D['H'])
reg, cx, cy, tx, ty, rad = D['reg'], D['cx'], D['cy'], D['tx'], D['ty'], D['rad']
n = Z.shape[1]; c0 = (n - 1) / 2.0; yy, xx = np.mgrid[0:n, 0:n]
# центр эталона в пикселях окна
TX = np.full(len(reg), np.nan); TY = np.full(len(reg), np.nan)
for rg in sorted(set(reg.tolist())):
    s = np.flatnonzero(reg == rg)
    crs = 'EPSG:' + json.load(open(glob.glob(f'/workspace/data/train-expds-3/{rg}/UTM.json')[0]))['crs'].split(':')[-1]
    a = transform('EPSG:3857', crs, np.r_[cx[s], tx[s]].tolist(), np.r_[cy[s], ty[s]].tolist())
    k = len(s); TX[s] = np.array(a[0][k:]) - np.array(a[0][:k]); TY[s] = np.array(a[1][k:]) - np.array(a[1][:k])
for TR in (4.0, 8.0):
    res = {v: [] for v in ('PK1', 'PK2', 'CM')}; before = []
    for p in np.flatnonzero(ok):
        z = Z[p]; m = np.isfinite(z)
        if m.mean() < 0.6 or not m[int(c0), int(c0)]: continue
        # заполнение пропусков ближайшим
        idx = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
        zf = z[tuple(idx)]
        l = zf - ndi.gaussian_filter(zf, TR / PX)
        before.append(np.hypot(TX[p], TY[p]))
        for v, sg in (('PK1', 1.0), ('PK2', 2.0)):
            zs = ndi.gaussian_filter(l, sg / PX); disk = (xx - c0) ** 2 + (yy - c0) ** 2 <= (3.0 / PX) ** 2
            i = np.argmax(np.where(disk, zs, -np.inf)); dx, dy = (i % n - c0) * PX, -(i // n - c0) * PX
            res[v].append(np.hypot(TX[p] - dx, TY[p] - dy))
        zs = ndi.gaussian_filter(l, 0.5 / PX); ro = float(np.clip(1.3 * rad[p], 2, 10)); px_, py_ = c0, c0
        for _ in range(3):
            r2 = (xx - px_) ** 2 + (yy - py_) ** 2
            ring = (r2 > (ro / PX) ** 2) & (r2 <= ((ro + 3) / PX) ** 2)
            w = np.clip(zs - np.median(zs[ring]), 0, None) * (r2 <= (ro / PX) ** 2)
            if w.sum() <= 0: break
            px_, py_ = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
            d = np.hypot(px_ - c0, py_ - c0) * PX
            if d > 3: px_, py_ = c0 + (px_ - c0) * 3 / d, c0 + (py_ - c0) * 3 / d
        dx, dy = (px_ - c0) * PX, -(py_ - c0) * PX
        res['CM'].append(np.hypot(TX[p] - dx, TY[p] - dy))
    b = np.array(before)
    for v, e in res.items():
        e = np.array(e)
        print(f'{sys.argv[1]} DEM TR={TR:g} {v} n {len(e)} ошибка до p50 {np.median(b):.2f} м; после p50 {np.median(e):.2f} p90 {np.percentile(e, 90):.2f} м; '
              f'ближе у {100 * np.mean(e < b - 1e-6):.0f}%; ≤1 м до {100*np.mean(b<=1):.0f}% после {100*np.mean(e<=1):.0f}%', flush=True)
