"""geo_snap (bot-3): LOFO-прогон P(object_poly) по участкам 200–217 (модель без семьи участка, pixgbm/models/<семья>/poly.txt),
из одной карты P — кандидаты: pole (как S, проверка воспроизведения), peaks r=0,5/1,0/2,0 м (P1/P2). Растры P не сохраняются."""
import sys, json, time
sys.path.insert(0, '/workspace/koz3_shared/bot-3/repo/scripts')
import pixgbm as X
from pathlib import Path
import lightgbm as lgb
G = X.G
R = '/workspace/data/train-expds-3'; MO = Path('/workspace/koz3_shared/bot-3/pixgbm/models'); W = Path('/workspace/koz3_shared/bot-3/geo_snap')
cfg = X.load_config(None); geo_cfg = cfg.get('geophys', {})
MODES = {'pole': dict(mode='pole'), 'pk05': dict(mode='peaks', peak_r_m=0.5), 'pk10': dict(mode='peaks', peak_r_m=1.0),
         'pk20': dict(mode='peaks', peak_r_m=2.0)}
feats = {k: [] for k in MODES}; stats = []
for reg in X.regions(R, sys.argv[1:] or None):
    fam = X.site_family(reg.name); bst = {'poly': lgb.Booster(model_file=str(MO / fam / 'poly.txt'))}
    crs_region = X.region_crs(reg, cfg)
    for mt in ('maggr', 'etgr', 'terfl'):
        files = G.pick_rasters(reg.rasters.get(mt, []))
        if not files: continue
        boundary = X.geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
        noise = X.geophys.load_mask(reg.noise, mt, crs_region)
        for f in files:
            t0 = time.time(); d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
            if not d['crs']: continue
            Pm = G.prob_map(d, mt, bst)
            rec = dict(region=reg.name, raster=f.name, mt=mt, model=fam)
            for k, kw in MODES.items():
                c = G.poly_candidates(Pm['poly'], d, reg.name, mt, f.name, 0.5, **kw); feats[k] += c; rec[k] = len(c)
            rec.update(sec=round(time.time() - t0, 1), maxrss_mb=round(X.rss_mb())); stats.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True); del d, Pm
tag = '_'.join(a[:3] for a in sys.argv[1:]) or 'all'
for k, v in feats.items():
    fc = X.writer.to_feature_collection(v)
    for ft, src in zip(fc['features'], v):
        for key in ('threshold_group', 'source'): ft['properties'][key] = src[key]
    (W / 'cand').mkdir(exist_ok=True); (W / 'cand' / f'{k}_{tag}.geojson').write_text(json.dumps(fc, ensure_ascii=False), encoding='utf-8')
(W / 'cand' / f'stats_{tag}.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1))
