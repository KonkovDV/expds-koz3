"""211: какая маска убирает 70 находок серой карты; где эталон относительно карт и масок."""
import sys, resource, json
resource.setrlimit(resource.RLIMIT_AS, (2_000_000_000,) * 2)
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
import logging; logging.disable(logging.WARNING)
from pathlib import Path
from shapely.geometry import shape
from shapely.ops import unary_union
from expds import geophys
from expds.catalog import scan_root
from expds.pipeline import load_config, region_crs
cfg = load_config(Path('/workspace/koz3_shared/bot-6/repo/configs/default.toml')); g = cfg.get('geophys', {})
print('boundary_min_cover', g.get('boundary_min_cover'), 'noise_max_cover', g.get('noise_max_cover'), 'uses_boundary', geophys.uses_survey_boundary('maggr', g))
reg = [r for r in scan_root(Path('/workspace/data/train-expds-3')) if r.name.startswith('211_')][0]
crs = region_crs(reg, cfg)
print('boundaries', [f.name for f in reg.boundaries], 'noise', [f.name for f in reg.noise])
B = geophys.survey_boundary(reg.boundaries, 'maggr', crs, g); N = geophys.load_mask(reg.noise, 'maggr', crs)
per = {f.name: geophys.load_mask([f], 'maggr', crs) for f in reg.noise}
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
G = [shape(f['geometry']) for f in gt['features'] if f['properties'].get('region_name', '').startswith('211')]
Gp = [x.representative_point() for x in G]
print('GT', len(G), 'в шуме (объединение)', sum(N.contains(p) for p in Gp) if N else None, 'внутри границы', sum(B.contains(p) for p in Gp) if B else None)
for k, m in per.items(): print('  шум', k, 'площадь м² (3857)', round(m.area), 'GT внутри', sum(m.contains(p) for p in Gp))
for path in reg.rasters['maggr']:
    fp = geophys.map_footprint(path, crs)
    b, n = geophys.sane_masks(path, B, N, g, crs)
    p, o = geophys.params_for(g, 'maggr', 'finds_points')
    found = geophys.finds_points(path, reg.name, 'maggr', p, fallback_crs=crs)
    if o['rank']: found = geophys.rank_within_source(found)
    print(f'{path.name}: рамка∩граница {fp.intersection(B).area/fp.area:.2f}, рамка∩шум {fp.intersection(N).area/fp.area:.2f}, GT в рамке {sum(fp.contains(q) for q in Gp)}; '
          f'находок {len(found)}: вне границы {sum(not b.contains(f["geom"].representative_point()) for f in found) if b else "-"}, в шуме {sum(n.contains(f["geom"].representative_point()) for f in found) if n else "-"}; '
          f'по файлам шума {[sum(m.contains(f["geom"].representative_point()) for f in found) for m in per.values()]}')
