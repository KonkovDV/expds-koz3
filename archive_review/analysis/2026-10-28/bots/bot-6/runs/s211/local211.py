"""211: попадания классики геофизики сдачи (pipeline.geophys_feats) при maps="gray" (как в сдаче) и maps="all".
Только участок 211, официальное сопоставление (expds.metric.build_curves); пороги min_confidence — из конфига сдачи.
Это локальный подсчёт tp/fp на трейне 211 (порог подбирался и на 211 тоже), не честный балл."""
import sys, json, copy, resource
resource.setrlimit(resource.RLIMIT_AS, (2_000_000_000,) * 2)
sys.path.insert(0, '/workspace/koz3_shared/bot-6/repo')
import logging; logging.disable(logging.WARNING)
from pathlib import Path
from shapely.geometry import mapping
from expds.catalog import scan_root
from expds.pipeline import load_config, geophys_feats, TimeGuard
from expds.metric import build_curves
cfg0 = load_config(Path('/workspace/koz3_shared/bot-6/repo/configs/default.toml'))
reg = [r for r in scan_root(Path('/workspace/data/train-expds-3')) if r.name.startswith('211_')][0]
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
gt = {'type': 'FeatureCollection', 'features': [f for f in gt['features'] if f['properties'].get('region_name', '').startswith('211')]}
for maps in ('gray', 'all'):
    cfg = copy.deepcopy(cfg0); cfg.setdefault('geophys', {}).setdefault('maggr', {})['maps'] = maps
    feats = geophys_feats(reg, cfg, TimeGuard(1e9))
    fc = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': mapping(f['geom']),
          'properties': {k: v for k, v in f.items() if k != 'geom' and not k.startswith('_')}} for f in feats]}
    cur = build_curves(fc, gt, '/workspace/expds-koz3/materials/compute_metrics.py')
    for cls, c in cur.items():
        if c.n_gt or len(c.conf):
            tp = int(c.hit.sum()); n = len(c.conf)
            print(f'maps={maps} {cls}: кандидатов {n}, tp {tp}, fp {n-tp}, GT {c.n_gt}, P {tp/(tp+0.5*(n-tp)) if n else 0:.3f}, R {tp/c.n_gt if c.n_gt else 0:.3f}')
