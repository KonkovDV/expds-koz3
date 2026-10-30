"""P-W2-10 шаг 1: дыры DEM (класс 2) как в prep c3 (vendor expds, Rust las_grid res 0.1, max_px 60M) — точно для локальных LAS,
оценка Пуассона exp(-λ) для всех 59 (λ = точек класса 2 на ячейку DEM; hsparity n·g2, площадь = valid_px·0.25 м²)."""
import sys, json, glob, math, time
import numpy as np
sys.path.insert(0, '/workspace/koz3/w2/vendor')
from expds import accel as XA, lidar as XL
from scipy import ndimage as ndi
W = '/workspace/koz3/w2'
LAS = {p.split('/')[-3]: p for p in glob.glob(f'{W}/las/*/*/*.las') + glob.glob(f'{W}/gaudit/las/*/*/*.las')}
REL = {p.split('/')[-1][:-5]: json.load(open(p)) for p in glob.glob(f'{W}/relief/cnx_relief/*.json') if 'manifest' not in p}
HS = {d['site']: d for d in json.load(open(f'{W}/notes/w2_hsparity_ci28_rows.json'))}
out = dict(backend=XA.backend(), exact={}, poisson={})
for site, p in sorted(LAS.items(), key=lambda kv: kv[0]):
    t = time.time()
    dem, info = XA.las_grid(p, 0.1, 2, 60_000_000)
    v = np.isfinite(dem); hull = XA.convex_hull_mask(v); hole = hull & ~v; res = info['res']
    lab, n = ndi.label(hole); sz = np.bincount(lab.ravel())[1:] * res * res if n else np.zeros(0)
    tot = hole.sum() * res * res
    big = {f'ge{a}m2': round(float(sz[sz >= a].sum() / max(tot, 1e-9)), 4) for a in (1, 10, 100)}
    nv = int(v.sum())
    al, _ = XA.las_grid(p, 0.5, 2, 60_000_000, use_all=True); g5, _ = XA.las_grid(p, 0.5, 2, 60_000_000)
    A5 = np.isfinite(al); H5 = A5 & ~np.isfinite(g5); l5, n5 = ndi.label(H5); s5 = np.bincount(l5.ravel())[1:] * 0.25 if n5 else np.zeros(0)
    e5 = dict(survey_cells05=int(A5.sum()), hole_frac05=round(float(H5.sum() / A5.sum()), 4),
              hole05_share_comp_ge10m2=round(float(s5[s5 >= 10].sum() / max(s5.sum(), 1e-9)), 4),
              hole05_share_comp_ge100m2=round(float(s5[s5 >= 100].sum() / max(s5.sum(), 1e-9)), 4), largest_hole05_m2=float(s5.max()) if n5 else 0.)
    out['exact'][site] = dict(dem_res=res, hull_cells=int(hull.sum()), valid_cells=nv, hole_frac_hull=round(float(hole.sum() / hull.sum()), 4),
        hole_area_m2=round(float(tot)), largest_hole_m2=round(float(sz.max()) if n else 0, 1), hole_area_share_in_comp=big,
        fill_method_now='tin(rust)' if nv <= 20_000_000 else 'pyramid', **e5, t_s=round(time.time() - t, 1))
    print(site, out['exact'][site], flush=True); del dem, v, hull, hole, lab
for site, m in sorted(REL.items()):
    h = HS.get(site)
    if not h: continue
    g = h['n'] * h['g2']; area = m['valid_px'] * 0.25; lam = g / area * m['dem_res'] ** 2
    l5 = g / area * 0.25; nvc = area / m['dem_res'] ** 2 * (1 - math.exp(-lam))
    out['poisson'][site] = dict(est_hole_frac05=round(math.exp(-l5), 4), est_valid_dem_cells=int(nvc), est_fill='tin' if nvc <= 20e6 else 'pyramid',dem_res=m['dem_res'], g2_frac=h['g2'], used_all=h['used_all'], pts_per_cell=round(lam, 2),
                                est_hole_frac=round(math.exp(-lam), 4))
json.dump(out, open(f'{W}/demfill/out/gapstats.json', 'w'), ensure_ascii=False, indent=1)
ps = out['poisson']; print('poisson05 >5%:', {k: v['est_hole_frac05'] for k, v in ps.items() if v['est_hole_frac05'] > 0.05}); print('pyramid:', [k for k, v in ps.items() if v['est_fill'] == 'pyramid'])
