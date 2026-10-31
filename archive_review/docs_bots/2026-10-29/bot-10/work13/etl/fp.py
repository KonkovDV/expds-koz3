"""Задача C: слои EtGr для классических кандидатов без source (по рамке растра), проверка рамок на эталоне. bot-3 venv (rasterio)."""
import rasterio, glob, json, re, collections
from shapely.geometry import shape, box
from shapely.ops import transform
from pyproj import Transformer
tr = Transformer.from_crs(3857, 32636, always_xy=True).transform
B = collections.defaultdict(list)
for f in sorted(glob.glob('/workspace/data/train-expds-3/21[2-7]_*/*EtGr*/*.tif')):
    reg = f.split('/')[-3]; key = re.sub(r'^\d+_', '', f.split('/')[-1][:-4])
    with rasterio.open(f) as d: B[reg].append((key, box(*d.bounds)))
def layers(reg, g, buf=0.5):
    p = transform(tr, g).representative_point()
    return [k for k, b in B.get(reg, []) if b.buffer(buf).contains(p)]
# эталон: попадает ли объект в рамку своего слоя
G = json.load(open('/workspace/rerank/gt_train_v2.geojson'))['features']
st = collections.Counter(); nl = []
for f in G:
    p = f['properties']
    if p['markup_type'] != 'etgr': continue
    key = re.sub(r'_ObjectPoly\.geojson$', '', p['source']); L = layers(p['region_name'], shape(f['geometry']))
    st['own_in' if key in L else 'own_out'] += 1; nl.append(len(L))
print('GT', dict(st), 'mean n layers', sum(nl)/len(nl), collections.Counter(nl).most_common(6))
out = open('Гнёздово.jsonl', 'w', encoding='utf-8'); c = collections.Counter()
for l in open('../../work11/fam0/Гнёздово.jsonl', encoding='utf-8'):
    f = json.loads(l); p = f['properties']
    if p.get('markup_type') == 'etgr' and not p.get('source'):
        L = layers(p['region_name'], shape(f['geometry'])); c[min(len(L), 9)] += 1
        p['etgr_layers'] = '|'.join(L) if L else '?'
    out.write(json.dumps(f, ensure_ascii=False) + '\n')
print('classic etgr n_layers', sorted(c.items()))
