"""211_Гнёздово_Терраса_3: почему в стеке 0 кандидатов. Шаги pipeline.geophys_feats по отдельности, конфиг сдачи."""
import sys, resource, json
resource.setrlimit(resource.RLIMIT_AS, (2_000_000_000,) * 2)
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
from pathlib import Path
import numpy as np, rasterio
from expds import geophys
from expds.catalog import scan_root
from expds.pipeline import load_config, region_crs
cfg = load_config(Path('/workspace/koz3_shared/bot-6/repo/configs/default.toml')) if True else None
root = Path('/workspace/data/train-expds-3')
reg = [r for r in scan_root(root) if r.name.startswith('211_')][0]
geo_cfg = cfg.get('geophys', {}); mt = 'maggr'
print('maps cfg:', (geo_cfg.get(mt) or {}).get('maps', 'gray'))
crs = region_crs(reg, cfg); print('crs', crs)
allf = reg.rasters.get(mt, []); print('rasters', [f.name for f in allf])
print('gray →', [f.name for f in geophys.select_rasters(allf, mt, 'gray')])
for path in allf:
    with rasterio.open(path) as ds:
        a = ds.read(); print(f'\n== {path.name}: {ds.width}x{ds.height} bands {ds.count} dtype {ds.dtypes[0]} crs {ds.crs} res {ds.res}')
        for b in range(a.shape[0]):
            x = a[b].astype(float); print(f'  band{b+1}: min {x.min():.0f} p1 {np.percentile(x,1):.0f} p50 {np.percentile(x,50):.0f} p99 {np.percentile(x,99):.0f} max {x.max():.0f} share==max {(x==x.max()).mean():.3f} share==255 {(x==255).mean():.3f}')
    boundary = geophys.survey_boundary(reg.boundaries, mt, crs, geo_cfg); noise = geophys.load_mask(reg.noise, mt, crs)
    b_map, n_map = geophys.sane_masks(path, boundary, noise, geo_cfg, crs)
    for cls in ('finds_points', 'object_poly'):
        p, o = geophys.params_for(geo_cfg, mt, cls)
        found = (geophys.finds_points(path, reg.name, mt, p, fallback_crs=crs) if cls == 'finds_points'
                 else geophys.object_polys(path, reg.name, mt, p, fallback_crs=crs))
        n0 = len(found)
        if o['rank']: found = geophys.rank_within_source(found)
        m = geophys.apply_masks(found, b_map, n_map if o['use_noise'] else None)
        k = [f for f in m if f['confidence'] >= o['min_confidence']]
        print(f'  {cls}: детектор {n0}; после масок {len(m)}; ≥min_conf {o["min_confidence"]}: {len(k)}')
