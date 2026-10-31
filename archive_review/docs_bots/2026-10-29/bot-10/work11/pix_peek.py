import json, collections, numpy as np
d = json.load(open('/workspace/koz3_shared/bot-10/in/geo_pix_S_train.geojson', encoding='utf-8'))
print(d.keys() if isinstance(d, dict) else type(d), len(d['features']))
keys = collections.Counter(tuple(sorted(f['properties'])) for f in d['features']); print(keys.most_common(3))
print(d['features'][0]['properties']); print(d['features'][0]['geometry']['type'])
C = collections.defaultdict(list); R = collections.defaultdict(set)
for f in d['features']:
    p = f['properties']; g = p.get('threshold_group') or p.get('markup_type'); C[(g, p.get('class_name'))].append(float(p.get('confidence', -1))); R[(g, p.get('class_name'))].add(p.get('region_name'))
for k, v in sorted(C.items()):
    v = np.asarray(v); print(k, len(v), len(R[k]), 'min', v.min().round(4), 'q', np.quantile(v, [.1, .5, .9]).round(4), 'max', v.max().round(4))
