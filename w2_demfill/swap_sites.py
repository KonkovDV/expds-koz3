"""P-W2-10: в стеке s012 (A + s012avg аддитивно) заменить li_cnx-строки только 4 участков на строки из merged OOF арма.
usage: swap_sites.py <stack.geojson> <merged_arm.geojson> <out.geojson> <fid_prefix> <reg1,reg2,...>"""
import json, sys, ijson, collections
st, new, out, pre, regs = sys.argv[1:6]; R = set(regs.split(','))
F = [f for f in json.load(open(new, encoding='utf-8'))['features'] if f['properties'].get('region_name') in R]
drop = collections.Counter(); add = collections.Counter(f['properties']['region_name'] for f in F); kept = 0
with open(out, 'w', encoding='utf-8') as o, open(st, 'rb') as fh:
    o.write('{"type": "FeatureCollection", "features": ['); first = True
    for f in ijson.items(fh, 'features.item', use_float=True):
        p = f.get('properties') or {}
        if p.get('threshold_group') == 'li_cnx' and p.get('region_name') in R:
            drop[p['region_name']] += 1; continue
        o.write(('' if first else ', ') + json.dumps(f, ensure_ascii=False)); first = False; kept += 1
    for i, f in enumerate(F):
        p = dict(f['properties']); assert p['threshold_group'] == 'li_cnx' and p['markup_type'] == 'li'; p['fid'] = f'{pre}_{i}'
        o.write(', ' + json.dumps({'type': 'Feature', 'geometry': f['geometry'], 'properties': p}, ensure_ascii=False))
    o.write(']}')
print(json.dumps(dict(kept=kept, dropped=dict(drop), added=dict(add)), ensure_ascii=False))
