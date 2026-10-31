"""Проверка правила «курганы внутри городищ не размечаются» на эталоне трейна (gt_train_v2)."""
import json, sys
from collections import defaultdict, Counter
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.strtree import STRtree
G = json.load(open('/workspace/rerank/gt_train_v2.geojson', encoding='utf-8'))['features']
by = defaultdict(lambda: defaultdict(list))
mk = Counter()
for f in G:
    p = f['properties']; c = p.get('class_name')
    by[p.get('region_name')][c].append((shape(f['geometry']), p.get('markup_type')))
KC = ('kurgany_tselye', 'kurgany_povrezhdennye')
tot_in = tot_k = tot_kr = 0; rows = []
for r, d in sorted(by.items()):
    gor = [g for g, _ in d.get('gorodishcha', [])]
    if not gor: continue
    U = unary_union([g.buffer(0) for g in gor])
    ks = [g for c in KC for g, _ in d.get(c, [])]
    n_in = sum(U.contains(g.centroid) for g in ks)
    tot_in += n_in; tot_k += len(ks); tot_kr += 1
    rows.append((r, len(gor), len(ks), n_in, sorted({m for _, m in d['gorodishcha']}), round(U.area)))
print('регионов с городищами', len(rows), 'курганов в них', tot_k, 'центроид внутри городища', tot_in)
for x in rows:
    if x[2] or x[3]: print(x)
print('гор. геометрии:', Counter(g.geom_type for r in by.values() for g, _ in r.get('gorodishcha', [])))
