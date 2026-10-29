"""Пункт 2: замер стоимости ансамбля 3 фолд-моделей против all (prob_map, признаки общие). Выход out/ens_timing.json."""
import sys, json, time, gc
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt/scripts')
import pixgbm as P, lightgbm as lgb
from pathlib import Path
G, geophys = P.G, P.geophys
MF = Path('/workspace/koz3_shared/bot-3/pixgbm/models'); FAMS = ['Гнёздово', 'Семибратнее', 'Судьбищи']
bst = lambda d: {k: lgb.Booster(model_file=str(d / f'{k}.txt')) for k in ('poly', 'finds')}
ALL = bst(MF / 'all'); FOLD = {f: bst(MF / f) for f in FAMS}
cfg = P.load_config(str(P.ROOT / 'configs' / 'default.toml')); geo_cfg = cfg.get('geophys', {})
res = []
for reg in P.regions('/workspace/data/train-expds-3', set(sys.argv[1:])):
    crs_region = P.region_crs(reg, cfg)
    for mt in ('maggr', 'etgr'):
        files = G.pick_rasters(reg.rasters.get(mt, []), None, mt, crs_region)[:1]
        if not files: continue
        boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg); noise = geophys.load_mask(reg.noise, mt, crs_region)
        d = G.load_map(files[0], G.GSD[mt], crs_region, boundary, noise)
        ks = ('poly', 'finds') if mt == 'maggr' else ('poly',)
        t = time.time(); G.prob_map(d, mt, {k: ALL[k] for k in ks}); ta = time.time() - t; gc.collect()
        t = time.time(); G.prob_map(d, mt, {f'{f}|{k}': FOLD[f][k] for f in FAMS for k in ks}); te = time.time() - t; gc.collect()
        res.append(dict(region=reg.name, mt=mt, npx=int(d['z'].size), sec_all=round(ta, 1), sec_ens3=round(te, 1), ratio=round(te / ta, 2), maxrss_mb=round(P.rss_mb())))
        print(json.dumps(res[-1], ensure_ascii=False), flush=True); del d; gc.collect()
json.dump(res, open('/workspace/koz3_shared/bot-3/audit/out/ens_timing.json', 'w'), ensure_ascii=False, indent=1)
