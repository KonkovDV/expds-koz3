"""Пункт 4: сколько трейн-кандидатов pix (pixgbm/pix_lofo.geojson) на участках «Гнёздово» гасило бы правило U
(вариант H) против стека geo_gap/sub_geo.pkl: старое (только класс) vs исправленное (класс+метод; слой у EtGr
неизвестен в стеке → верхняя граница). → out/union_affected.json"""
import json, pickle, collections, shapely
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.ops import transform
from pyproj import Transformer
TR = {}
# стек в UTM участка, pix в EPSG:3857 → центр pix переводится в original_crs
S = pickle.load(open('/workspace/koz3_shared/bot-3/geo_gap/sub_geo.pkl', 'rb'))
fc = json.load(open('/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson'))
H = lambda r: 'Гнёздово' in r
res = {}
for c in ('object_poly', 'finds_points'):
    s = S[c]; g = shapely.from_wkb(s['wkb'])
    base = [(g[i], str(s['reg'][i]), str(s['grp'][i])) for i in range(len(g)) if H(str(s['reg'][i]))]
    tr = STRtree([b[0] for b in base])
    cnt = collections.Counter()
    for f in fc['features']:
        p = f['properties']
        if p['class_name'] != c or not H(p['region_name']): continue
        oc = p['original_crs']; t = TR.setdefault(oc, Transformer.from_crs('EPSG:3857', oc, always_xy=True))
        ce = transform(t.transform, shape(f['geometry'])).centroid; mt = p['markup_type']; cnt[f'pix_{mt}'] += 1
        q = ce if c == 'object_poly' else ce.buffer(1.0)
        hits = [base[int(j)] for j in tr.query(q) if base[int(j)][1] == p['region_name'] and
                (base[int(j)][0].contains(ce) if c == 'object_poly' else base[int(j)][0].centroid.distance(ce) <= 1.0)]
        if not hits: continue
        cnt[f'old_drop_{mt}'] += 1
        if any(h[2] == mt for h in hits): cnt[f'new_drop_{mt}_upper'] += 1
        else: cnt[f'cross_type_only_{mt}'] += 1
    res[c] = dict(sorted(cnt.items()))
print(json.dumps(res, ensure_ascii=False))
json.dump(res, open('out/union_affected.json', 'w'), ensure_ascii=False, indent=1)
