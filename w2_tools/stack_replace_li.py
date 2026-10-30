"""W2 P-W2-6 вариант (b): стек A без li-слоя (markup_type == "li") + OOF cnx (threshold_group li_cnx).
usage: stack_replace_li.py <stack_arch.geojson> <oof_cnx.geojson> <out.geojson> <fid_prefix> [--classes c1,c2]
--classes: удалять/дописывать только эти классы (b-kur); по умолчанию все. Потоково (ijson), conf и геометрия без изменений."""
import json, sys, ijson, collections
st, oof, out, pre = sys.argv[1:5]
CLS = set(sys.argv[sys.argv.index('--classes') + 1].split(',')) if '--classes' in sys.argv else None
def sel(p): return CLS is None or p.get('class_name') in CLS
F = [f for f in json.load(open(oof, encoding='utf-8'))['features'] if sel(f['properties'])]
drop = collections.Counter(); kept = 0
with open(out, 'w', encoding='utf-8') as o, open(st, 'rb') as fh:
    o.write('{"type": "FeatureCollection", "features": ['); first = True
    for f in ijson.items(fh, 'features.item', use_float=True):
        p = f.get('properties') or {}
        if p.get('markup_type') == 'li' and sel(p):
            drop[(p.get('threshold_group'), p.get('class_name'))] += 1; continue
        o.write(('' if first else ', ') + json.dumps(f, ensure_ascii=False)); first = False; kept += 1
    for i, f in enumerate(F):
        p = dict(f['properties']); assert p['threshold_group'] == 'li_cnx' and p['markup_type'] == 'li'; p['fid'] = f'{pre}_{i}'
        o.write(', ' + json.dumps({'type': 'Feature', 'geometry': f['geometry'], 'properties': p}, ensure_ascii=False))
    o.write(']}')
print(json.dumps(dict(kept=kept, dropped=sum(drop.values()), added=len(F), drop={f'{g}|{c}': n for (g, c), n in sorted(drop.items(), key=str)}), ensure_ascii=False))
