"""211: значения серой и цветной карты внутри границы съёмки и в точках эталона."""
import sys, json, resource
resource.setrlimit(resource.RLIMIT_AS, (2_000_000_000,) * 2)
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
import logging; logging.disable(logging.WARNING)
from pathlib import Path
import numpy as np, rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape, mapping
from shapely.ops import transform as stf
from pyproj import Transformer
from expds import geophys
from expds.catalog import scan_root
from expds.pipeline import load_config, region_crs
cfg = load_config(Path('/workspace/koz3_shared/bot-6/repo/configs/default.toml')); g = cfg.get('geophys', {})
reg = [r for r in scan_root(Path('/workspace/data/train-expds-3')) if r.name.startswith('211_')][0]
crs = region_crs(reg, cfg)
B = geophys.survey_boundary(reg.boundaries, 'maggr', crs, g)
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
G = [shape(f['geometry']).representative_point() for f in gt['features'] if f['properties'].get('region_name', '').startswith('211') and f['properties']['class_name'] == 'finds_points']
for path in reg.rasters['maggr']:
    with rasterio.open(path) as ds:
        t = Transformer.from_crs('EPSG:3857', ds.crs, always_xy=True).transform
        Bl = stf(t, B); a = ds.read(1).astype(float)
        inside = ~geometry_mask([mapping(Bl)], out_shape=a.shape, transform=ds.transform)
        v = a[inside]
        rc = [ds.index(*t(p.x, p.y)) for p in G]
        gv = np.array([a[r, c] for r, c in rc if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]])
        print(f'{path.name}: внутри границы {inside.sum()} px; band1 p5/p50/p95 {np.percentile(v,[5,50,95]).round()} доля 255 {(v==255).mean():.3f} доля 0 {(v==0).mean():.3f} уник {len(np.unique(v))}; '
              f'в точках эталона ({len(gv)}): доля 255 {(gv==255).mean():.3f}, p5/p50/p95 {np.percentile(gv,[5,50,95]).round()}; вне границы доля 255 {(a[~inside]==255).mean():.3f}')
