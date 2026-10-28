# Дубли в эталоне: пары эталонов одного класса/региона с IoU>0.5 или cov(min)>=0.5 и общим rp
from common7 import *
from shapely.geometry import shape
from shapely import STRtree
from collections import defaultdict
gt = load('/workspace/rerank/gt_train_v2.geojson')
by = defaultdict(list)
for g in gt['features']:
    p = g['properties']; by[(p['region_name'], p['class_name'])].append(shape(g['geometry']))
res = defaultdict(lambda: [0, 0, 0, 0])
for (r, c), gs in by.items():
    gs = [g for g in gs if g.area > 0]
    t = STRtree(gs); res[c][0] += len(gs)
    inpair = set()
    for a, g in enumerate(gs):
        for b in t.query(g):
            b = int(b)
            if b <= a: continue
            h = gs[b]; i = g.intersection(h).area
            if i / (g.area + h.area - i) > 0.5: res[c][1] += 1; inpair |= {a, b}
            if i / min(g.area, h.area) >= 0.5: res[c][2] += 1
    res[c][3] += len(inpair)
print('class n_gt pairs_IoU>0.5 pairs_covmin>=0.5 gt_in_IoU_pairs')
for c, v in res.items(): print(c, *v)
