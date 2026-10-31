"""Привязка центров кандидатов к рельефу по карте Li_c (цветной локальный рельеф, синий=низ, красный=верх).
Запуск: /tmp/dfine_venv/bin/python snap_li.py cand.npz snap_li.npz
cand.npz: idx, reg, cx, cy (EPSG:3857), rad (радиус фигуры в м по земле).
Выход: idx, has (есть растр и покрытие), для каждого варианта dx_<V>, dy_<V> в метрах EPSG:3857.
Варианты:
  PK1: argmax сглаженного (σ=1 м) рельефа в круге 3 м вокруг центра;
  PK2: то же при σ=2 м;
  CM:  центр масс положительного рельефа (рельеф − медиана кольца [ro; ro+3]) в круге ro=clip(1.3·rad, 2, 10), 3 итерации mean-shift, сдвиг ≤ 3 м."""
import sys, json, glob, numpy as np, rasterio, cv2
from rasterio.windows import from_bounds
from rasterio.warp import transform
from rasterio.enums import Resampling
from scipy import ndimage as ndi
D = np.load(sys.argv[1], allow_pickle=True)
idx, reg, cx, cy, rad = D['idx'], D['reg'], D['cx'], D['cy'], D['rad']
PX = 0.2  # м на пиксель после передискретизации
TR = float(__import__('os').environ.get('SNAP_TR', 6.0))  # σ тренда, м
V = ['PK1', 'PK2', 'CM']
out = {f'{a}_{v}': np.zeros(len(idx)) for v in V for a in ('dx', 'dy')}
has = np.zeros(len(idx), bool); stats = {}
def proxy(a):
    im = np.ascontiguousarray(np.moveaxis(a, 0, -1)); m = im.sum(-1) > 0
    h = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[..., 0].astype(np.float32)
    h = np.where(h >= 150, h - 180, h)          # пурпурно-красный -> выше красного
    return (120.0 - h) / 150.0, m
for rg in sorted(set(reg.tolist())):
    ps = glob.glob(f'/workspace/data/train-expds-3/{rg}/02_*_Li_карты/*_Li_c.tif')
    if not ps: continue
    crs = json.load(open(glob.glob(f'/workspace/data/train-expds-3/{rg}/UTM.json')[0]))['crs']
    crs = 'EPSG:' + crs.split(':')[-1]
    sel = np.flatnonzero(reg == rg)
    X, Y = transform('EPSG:3857', crs, cx[sel].tolist(), cy[sel].tolist())
    n_ok = 0
    with rasterio.open(ps[0]) as ds:
        for k, p in enumerate(sel):
            x, y = X[k], Y[k]; ro = float(np.clip(1.3 * rad[p], 2, 10))  # rad — по земле
            H = ro + 3 + 15
            w = from_bounds(x - H, y - H, x + H, y + H, ds.transform)
            n = int(round(2 * H / PX))
            try:
                a = ds.read(window=w, out_shape=(3, n, n), resampling=Resampling.bilinear, boundless=True, fill_value=0)
            except Exception:
                continue
            z, m = proxy(a)
            zf = np.where(m, z, np.median(z[m]) if m.any() else 0)
            z = zf - ndi.gaussian_filter(zf, TR / PX)  # локальный рельеф (LRM): минус тренд
            c0 = (n - 1) / 2.0
            if not m[int(c0), int(c0)] or m.mean() < 0.8: continue
            yy, xx = np.mgrid[0:n, 0:n]
            res = {}
            for v, sg in (('PK1', 1.0), ('PK2', 2.0)):
                zs = ndi.gaussian_filter(np.where(m, z, np.median(z[m])), sg / PX)
                disk = (xx - c0) ** 2 + (yy - c0) ** 2 <= (3.0 / PX) ** 2
                i = np.argmax(np.where(disk, zs, -np.inf)); res[v] = ((i % n - c0) * PX, -(i // n - c0) * PX)
            zs = ndi.gaussian_filter(np.where(m, z, np.median(z[m])), 0.5 / PX)
            px, py = c0, c0
            for _ in range(3):
                r2 = (xx - px) ** 2 + (yy - py) ** 2
                ring = (r2 > (ro / PX) ** 2) & (r2 <= ((ro + 3) / PX) ** 2) & m
                if ring.sum() < 20: break
                wgt = np.clip(zs - np.median(zs[ring]), 0, None) * ((r2 <= (ro / PX) ** 2) & m)
                if wgt.sum() <= 0: break
                px, py = (wgt * xx).sum() / wgt.sum(), (wgt * yy).sum() / wgt.sum()
            dx, dy = (px - c0) * PX, -(py - c0) * PX
            s = np.hypot(dx, dy)
            if s > 3: dx, dy = dx * 3 / s, dy * 3 / s
            res['CM'] = (dx, dy)
            # обратно в EPSG:3857
            for v, (ddx, ddy) in res.items():
                bx, by = transform(crs, 'EPSG:3857', [x + ddx], [y + ddy])
                out[f'dx_{v}'][p] = bx[0] - cx[p]; out[f'dy_{v}'][p] = by[0] - cy[p]
            has[p] = True; n_ok += 1
    stats[rg] = [len(sel), n_ok]; print(rg, len(sel), n_ok, flush=True)
np.savez(sys.argv[2], idx=idx, has=has, **out)
for v in V:
    s = np.hypot(out[f'dx_{v}'], out[f'dy_{v}'])[has]
    print(v, 'n', has.sum(), 'shift m(3857) p50 %.2f p90 %.2f' % (np.median(s), np.percentile(s, 90)))
