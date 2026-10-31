"""Пункт 2: прогноз кандидатов на трейне сразу несколькими моделями (признаки окна считаются один раз).
Для участка семьи X:
  single_<H>  — модель одной семьи H≠X (вложенный OOF и C2);   ens2 — среднее P двух single-моделей (C2);
  ins_<F>     — fold-модель M_{−F} (pixgbm/models/<F>), F≠X — in-sample (C1);  all — модель all in-sample (сдвиг).
Выход: audit/calib/pred_<ключ>.geojson (+ .stats.json). Код кандидатов — как cmd_predict (t_pix 0,5)."""
import sys, json, time, gc
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt/scripts')
import pixgbm as P
import lightgbm as lgb
import numpy as np
from pathlib import Path
from expds.naming import site_family
G, geophys, writer = P.G, P.geophys, P.writer
ROOT = '/workspace/data/train-expds-3'
MS = Path('/workspace/koz3_shared/bot-3/audit/calib/models_single'); MF = Path('/workspace/koz3_shared/bot-3/pixgbm/models')
OUT = Path('/workspace/koz3_shared/bot-3/audit/calib')
FAMS = ['Гнёздово', 'Семибратнее', 'Судьбищи']
def bst(d):
    return {k: lgb.Booster(model_file=str(d / f'{k}.txt')) for k in ('poly', 'finds') if (d / f'{k}.txt').exists()}
SINGLE = {f: bst(MS / f) for f in FAMS}; FOLD = {f: bst(MF / f) for f in FAMS}; ALL = bst(MF / 'all')
cfg = P.load_config(str(P.ROOT / 'configs' / 'default.toml')); geo_cfg = cfg.get('geophys', {})
feats, stats = {}, []
only = set(sys.argv[1:]) or None
for reg in P.regions(ROOT, only):
    X = site_family(reg.name); others = [f for f in FAMS if f != X]
    sets = {**{f'single_{h}': SINGLE[h] for h in others}, **{f'ins_{f}': FOLD[f] for f in others}, 'all': ALL}
    crs_region = P.region_crs(reg, cfg)
    for mt in ('maggr', 'etgr', 'terfl'):
        files = G.pick_rasters(reg.rasters.get(mt, []), None, mt, crs_region)
        if not files: continue
        boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
        noise = geophys.load_mask(reg.noise, mt, crs_region)
        for f in files:
            t0 = time.time()
            d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
            if not d['crs']: continue
            use = {f'{s}|{k}': b for s, bs in sets.items() for k, b in bs.items() if k == 'poly' or mt == 'maggr'}
            Pm = {}
            npx = d['z'].size
            chunk = max(1, int(6e8 // (npx * 4)))                 # ≤ ~600 МБ карт за раз
            keys = list(use)
            for i in range(0, len(keys), chunk):
                Pm.update(G.prob_map(d, mt, {k: use[k] for k in keys[i:i + chunk]}))
            for s1, s2 in [(f'single_{others[0]}', f'single_{others[1]}')]:
                for k in ('poly', 'finds'):
                    if f'{s1}|{k}' in Pm and f'{s2}|{k}' in Pm:
                        Pm[f'ens2|{k}'] = (Pm[f'{s1}|{k}'] + Pm[f'{s2}|{k}']) / 2
            rec = dict(region=reg.name, raster=f.name, mt=mt)
            for key in sorted({k.split('|')[0] for k in Pm}):
                fp = G.poly_candidates(Pm[f'{key}|poly'], d, reg.name, mt, f.name, G.T_PIX) if f'{key}|poly' in Pm else []
                ff = G.finds_candidates(Pm[f'{key}|finds'], d, reg.name, mt, f.name) if f'{key}|finds' in Pm else []
                fk = key if not key.startswith('single_') else key   # имя набора
                feats.setdefault(fk, []).extend(fp + ff); rec[key] = [len(fp), len(ff)]
            rec.update(sec=round(time.time() - t0, 1), maxrss_mb=round(P.rss_mb()))
            stats.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
            del d, Pm; gc.collect()
for key, fs in feats.items():
    fc = writer.to_feature_collection(fs)
    for feat, src in zip(fc['features'], fs):
        for kk in ('threshold_group', 'source'): feat['properties'][kk] = src[kk]
    (OUT / f'pred_{key}.geojson').write_text(json.dumps(fc, ensure_ascii=False), encoding='utf-8')
(OUT / 'pred_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=1))
print('DONE', {k: len(v) for k, v in feats.items()}, flush=True)
