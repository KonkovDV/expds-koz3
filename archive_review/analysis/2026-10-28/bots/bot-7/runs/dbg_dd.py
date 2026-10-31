# Почему dedup 1 м теряет TP: сравнение на всём стеке с порогами in-sample
from common7 import *
from eval_geom import *
from expds.metric import apply_group_thresholds
st = load(W / 'stack_sub6000.geojson'); gt = load('/workspace/rerank/gt_train_v2.geojson')
curves = build_curves(st, gt, METRIC)
res = sweep_groups(curves); taus = {}
for c, v in res.items():
    for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
kfc = apply_group_thresholds(st, taus, METRIC); del st
C = 'kurgany_povrezhdennye'
kfc['features'] = [f for f in kfc['features'] if f['properties']['class_name'] == C]
gt['features'] = [g for g in gt['features'] if g['properties']['class_name'] == C]
ints = to_internal(kfc)
for i, d in enumerate(ints): d['_i'] = i
dd = geom_opt.dedup_anchors(ints, kur({'radius_m': 1.0}))
keep = {d['_i'] for d in dd}
hc = build_curves(kfc, gt, METRIC)[C]
print('kept', len(ints), 'after dd', len(dd), 'base tp', int(hc.hit.sum()))
# соответствие кривой и объектов: по (region, conf) — порядок в кривой может отличаться
from collections import Counter
hit_by = Counter(); tot_by = Counter()
import numpy as np
key = [(r, round(float(c), 6)) for r, c in zip(hc.pred_region, hc.conf)]
hits = Counter(k for k, h in zip(key, hc.hit) if h)
drop_hits = 0; groups = Counter(); nd = 0
for i, d in enumerate(ints):
    if i in keep: continue
    nd += 1
    k = (d['region_name'], round(float(d['confidence']), 6))
    if hits.get(k, 0) > 0: drop_hits += 1; groups[d.get('threshold_group') or d.get('markup_type')] += 1
print('dropped', nd, 'of which (by region,conf) were TP', drop_hits, groups)
fc2 = {'type': 'FeatureCollection', 'features': [f for i, f in enumerate(kfc['features']) if i in keep]}
hc2 = build_curves(fc2, gt, METRIC)[C]
print('dd tp', int(hc2.hit.sum()))
