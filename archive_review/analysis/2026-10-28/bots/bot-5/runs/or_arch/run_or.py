"""Пилот Or-архитектуры: готовые сети архитектуры (без обучения) на ортофото, шаг модели 0,5 м.
Тот же путь, что в боевом конвейере: expds.pipeline._raster_with_model (окно 4096, перекрытие 256).
Выход: признаки с conf (порог выдачи 0.05 — дальше порог задаётся при оценке).
usage: run_or.py MODEL.pt TIF REGION OUT.json
"""
import sys, json, time, resource
from pathlib import Path
ROOT = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(ROOT))
import torch; torch.set_num_threads(2)
from expds.infer_nn import Segmenter
from expds.pipeline import _raster_with_model
model, tif, region, out = sys.argv[1:5]
t0 = time.time()
seg = Segmenter(model, device='cpu')
print('model', model, 'res', seg.res, 'classes', seg.classes, 'meta tile', seg.meta.get('tile'), flush=True)
feats = _raster_with_model(seg, Path(tif), region, 'or', min_conf={c: 0.05 for c in seg.classes})
for f in feats:
    if 'geom' in f and hasattr(f['geom'], '__geo_interface__'):
        f['geom'] = f['geom'].__geo_interface__
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
json.dump({'model': model, 'tif': tif, 'region': region, 'n': len(feats), 'sec': time.time() - t0,
           'peak_rss_MB': rss, 'feats': feats}, open(out, 'w'), default=str)
print('n feats', len(feats), 'sec %.1f' % (time.time() - t0), 'peak_rss_MB %.0f' % rss, flush=True)
