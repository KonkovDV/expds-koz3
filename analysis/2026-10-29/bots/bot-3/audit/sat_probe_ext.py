"""Item 1 (диагностика, bot-3, 29.10): насыщение и выбор копии карты, как их видит PIXGBM (load_map + граница + шум,
ровно как scripts/pixgbm.py extract). Эталон — по ключу разметки (как extract: (метод, класс, label_key) ↔ raster_key),
а не «все находки участка» (у geo_sat_probe.py 211 серая и цвет_2 получают одну и ту же разметку).
Для КАЖДОЙ копии (и выбранной, и нет): доля эталона в ядрах ≤60 м² (satc), в любом насыщении (sat), ≤0,5 м от
насыщения, вне данных; hp_energy (без эталона, как в аудите); AUC сигнала |z − среднее 2 м| (макс. ≤0,3 м) в точках
эталона против случайных пикселей данных — какая копия несёт сигнал (диагностика с эталоном, не правило)."""
import sys, json, time, resource
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt')
import numpy as np
from scipy import ndimage as ndi
from expds import geo_pixgbm as G, geophys
from expds.catalog import scan_root
from expds.gt import read_annotation
from expds.pipeline import load_config, region_crs
from expds.crs import WEB_MERCATOR, to_crs
from shapely.ops import polylabel
from rasterio import features as rf

cfg = load_config('/workspace/koz3_shared/bot-3/audit/wt/configs/default.toml'); geo_cfg = cfg.get('geophys', {})
OUT = sys.argv[1]
rng = np.random.default_rng(0)

def hp_energy(d):
    z, v = d['z'], d['valid']
    k = max(3, int(round(2.0 / d['gsd'])) | 1)
    hp = np.abs(z - ndi.uniform_filter(np.where(v, z, 0), k))
    return hp, round(float(np.median(hp[v])), 4) if v.sum() > 100 else 0.0

def auc(pos, neg):
    if not len(pos) or not len(neg): return None
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float)
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))), 4)

rows = []
for reg in scan_root('/workspace/data/train-expds-3'):
    if not (reg.name[:3].isdigit() and 200 <= int(reg.name[:3]) <= 217): continue
    crs_region = region_crs(reg, cfg)
    anns = {}
    for f, cls, mt in reg.annotations:
        anns.setdefault((mt, cls, G.label_key(f)), []).append(f)
    for mt in ('maggr', 'etgr', 'terfl'):
        allf = reg.rasters.get(mt, [])
        if not allf: continue
        picked = set(G.pick_rasters(allf)); nkeys = len(picked)
        boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
        noise = geophys.load_mask(reg.noise, mt, crs_region)
        for f in sorted(allf):
            t0 = time.time(); key = G.raster_key(f)
            gt = {}
            for cls in ('object_poly', 'finds_points'):
                fs = anns.get((mt, cls, key))
                if not fs and nkeys == 1:
                    cand = [v for (m2, c2, _k), v in anns.items() if m2 == mt and c2 == cls]
                    fs = cand[0] if len(cand) == 1 else None
                gt[cls] = [] if not fs else [o['geom'] for p in fs for o in read_annotation(p, cls, mt, reg)]
            d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise, sat_core_max_m2=60)
            rec = dict(region=reg.name, mt=mt, file=f.name, key=key, picked=f in picked, gsd=d['gsd'], crs=d['crs'])
            if not d['crs']:
                rows.append(rec); continue
            v1, satc, sat = d['valid'], d['satc'], d['sat']
            v0 = v1 & ~satc                                # как в S (sat_core 0); граница/шум одинаковы
            H, W = v1.shape; inv = ~d['T']
            near = ndi.binary_dilation(sat, iterations=max(1, int(round(0.5 / d['gsd']))))
            rec.update(valid0_share=round(float(v0.mean()), 4), satc_share_of_valid=round(float(satc.sum() / max(v1.sum(), 1)), 5),
                       sat_share=round(float(sat.mean()), 4))
            hp, rec['hp_energy'] = hp_energy(dict(z=d['z'], valid=v0, gsd=d['gsd']))
            hpmax = ndi.maximum_filter(np.where(v0, hp, 0), max(3, int(round(0.6 / d['gsd'])) | 1))
            for k, cls in (('finds', 'finds_points'), ('poly', 'object_poly')):
                gg = [to_crs(g, WEB_MERCATOR, d['crs']) for g in gt[cls]]
                pts = []
                for g in gg:
                    if k == 'poly' and g.geom_type in ('Polygon', 'MultiPolygon'):
                        try: c = polylabel(g if g.geom_type == 'Polygon' else max(g.geoms, key=lambda x: x.area), tolerance=0.02)
                        except Exception: c = g.representative_point()
                    else: c = g.centroid
                    col, row = inv * (c.x, c.y); pts.append((int(np.floor(row)), int(np.floor(col))))
                rc = np.array(pts, int).reshape(-1, 2)
                ok = (rc[:, 0] >= 0) & (rc[:, 0] < H) & (rc[:, 1] >= 0) & (rc[:, 1] < W); rc = rc[ok]
                r_, c_ = rc[:, 0], rc[:, 1]
                inb = v1[r_, c_] | sat[r_, c_]            # внутри данных или в насыщении
                rec[f'gt_{k}'] = int(len(gg)); rec[f'gt_{k}_in_frame'] = int(len(rc))
                rec[f'gt_{k}_in_satc'] = int(satc[r_, c_].sum())
                rec[f'gt_{k}_in_sat'] = int(sat[r_, c_].sum())
                rec[f'gt_{k}_near_sat05'] = int((near[r_, c_] & ~sat[r_, c_]).sum())
                rec[f'gt_{k}_invalid_S'] = int((~v0[r_, c_]).sum())
                if k == 'poly' and gg:
                    m = rf.rasterize([(g, i + 1) for i, g in enumerate(gg)], out_shape=(H, W), transform=d['T'], all_touched=False)
                    lab = np.arange(1, len(gg) + 1)
                    npx = ndi.sum_labels(np.ones_like(m), m, lab); nsat = ndi.sum_labels(sat.astype(np.int32), m, lab)
                    nsatc = ndi.sum_labels(satc.astype(np.int32), m, lab)
                    has = npx > 0
                    rec['gt_poly_rasterized'] = int(has.sum())
                    rec['gt_poly_area_share_sat'] = round(float(nsat[has].sum() / max(npx[has].sum(), 1)), 4)
                    rec['gt_poly_area_share_satc'] = round(float(nsatc[has].sum() / max(npx[has].sum(), 1)), 4)
                    rec['gt_poly_ge50_sat'] = int(((nsat / np.maximum(npx, 1)) >= 0.5)[has].sum())
                    del m
                if k == 'finds' and len(rc) and v0.sum() > 1000:
                    vv = np.flatnonzero(v0.ravel()); neg = hpmax.ravel()[rng.choice(vv, min(20000, vv.size), replace=False)]
                    rec['finds_signal_auc'] = auc(hpmax[r_, c_][inb], neg)
            rec['sec'] = round(time.time() - t0, 1); rec['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            rows.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
            del d, v1, v0, satc, sat, near, hp, hpmax
json.dump(rows, open(OUT, 'w'), ensure_ascii=False, indent=1)
