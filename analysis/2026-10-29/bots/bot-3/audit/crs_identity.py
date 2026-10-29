"""Пункт 3: на трейне включение ground_metres ничего не меняет (все карты UTM) — побайтная проверка
load_map (z, zd, valid, noise, T, gsd) и geophys.read_band (массив, шаг) для всех выбранных карт геофизики."""
import sys, json, hashlib
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt/scripts')
import pixgbm as P
import numpy as np
from expds import crs as C
G, geophys = P.G, P.geophys
cfg = P.load_config(str(P.ROOT / 'configs' / 'default.toml')); geo_cfg = cfg.get('geophys', {})
h = lambda *a: hashlib.sha1(b''.join(x.tobytes() if hasattr(x, 'tobytes') else repr(x).encode() for x in a)).hexdigest()
rows, bad = [], 0
for reg in P.regions('/workspace/data/train-expds-3'):
    crs_region = P.region_crs(reg, cfg)
    for mt in ('maggr', 'etgr', 'terfl'):
        files = G.pick_rasters(reg.rasters.get(mt, []), None, mt, crs_region)
        if not files: continue
        boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
        noise = geophys.load_mask(reg.noise, mt, crs_region)
        for f in files:
            r = {}
            for on in (False, True):
                C.GROUND_METRES = on
                d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
                a, T, crs, res = geophys.read_band(f, 'first', crs_region)
                r[on] = (h(d['z'], d['zd'], d['valid'], d['noise'], tuple(d['T']), d['gsd']), h(np.nan_to_num(a, nan=-9e9), res), d.get('k'), d['crs'])
                del d, a
            same = r[False][:2] == r[True][:2]
            bad += not same
            rows.append(dict(region=reg.name, raster=f.name, mt=mt, crs=r[True][3], k_on=r[True][2], identical=same))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
json.dump(dict(n=len(rows), not_identical=bad, rows=rows), open('/workspace/koz3_shared/bot-3/audit/out/crs_identity.json', 'w'), ensure_ascii=False, indent=1)
print('n', len(rows), 'not identical', bad)
