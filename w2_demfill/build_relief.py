"""P-W2-10: рельеф c3 (lrm_s, lrm_l, slope) как build_relief prep-ядра, но с выбором lidar.dem_fill. Код expds = wt/w2 (флаг), Rust .so = вендор ядер.
usage: build_relief.py <site> <delaunay|tin_lowveg> <out_dir>"""
import sys, json, glob, time
sys.path[:0] = ['/workspace/koz3/wt/w2', '/workspace/koz3/w2/vendor']
import numpy as np, types
# rasterio в venv бокса нет; relief_channels берёт из него только from_origin для transform (на значения не влияет)
_r = types.ModuleType('rasterio'); _rt = types.ModuleType('rasterio.transform')
_rt.from_origin = lambda w, n, xs, ys: (w, n, xs, ys); _r.transform = _rt
sys.modules.setdefault('rasterio', _r); sys.modules.setdefault('rasterio.transform', _rt)
from expds import lidar as XL, accel as XA
site, fill, od = sys.argv[1:4]
W = '/workspace/koz3/w2'
las = [p for p in glob.glob(f'{W}/las/*/*/*.las') + glob.glob(f'{W}/gaudit/las/*/*/*.las') if p.split('/')[-3] == site][0]
km = json.load(open(f'{W}/relief/cnx_relief/{site}.json'))
t = time.time()
d = XL.build_dem(las, res=0.1, fill=fill, crs=km['crs'])
base, tr = XL.relief_channels(d, 0.5, ("lrm_s", "lrm_l", "slope"))
assert list(base.shape[1:]) == km['shape'][1:] and abs(d.left - km['left']) < 1e-6 and abs(d.top - km['top']) < 1e-6, (base.shape, km['shape'])
np.save(f'{od}/{site}.npy', base.astype(np.uint8))
k = np.load(f'{W}/relief/cnx_relief/{site}.npy', mmap_mode='r')[:3]
diff = {t_: dict(neq_frac=round(float((base[i] != k[i]).mean()), 6), mad=round(float(np.abs(base[i].astype(int) - k[i]).mean()), 4)) for i, t_ in enumerate(("lrm_s", "lrm_l", "slope"))}
meta = dict(km, shape=list(base.shape), dem_fill=fill, dem_res=float(d.res), backend=XA.backend(), valid_px=int((base[0] > 0).sum()),
            holes=d.info.get('holes'), lowveg=d.info.get('lowveg'), vs_kaggle_npy=diff, t_s=round(time.time() - t, 1))
json.dump(meta, open(f'{od}/{site}.json', 'w'), ensure_ascii=False, indent=1)
print(site, fill, json.dumps({k_: meta[k_] for k_ in ('dem_res', 'backend', 'valid_px', 'holes', 'lowveg', 'vs_kaggle_npy', 't_s')}, ensure_ascii=False), flush=True)
