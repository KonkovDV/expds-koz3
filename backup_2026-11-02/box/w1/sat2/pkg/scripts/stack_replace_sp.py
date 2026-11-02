"""KOZ3 W1 sat2 REPLACE (реконструкция bot-W1-aeor 01.11; оригинал bot2 на бокс не попал).
usage: stack_replace_sp.py <stack_arch.geojson> <oof_sp_det2.geojson> <out.geojson>
Стек A без курганов спутникового слоя (markup_type == "sp_or", классы kurgany_*; группы sp_focal/sp_focal2),
плюс все OOF sp_det2 (threshold_group sp_det2). sp_arch (архитектуры) и sp_or городища остаются. Потоково (ijson),
conf и геометрия без изменений — по образцу W2 stack_replace_li.py."""
import json, sys, ijson, collections
st, oof, out = sys.argv[1:4]
KUR = {'kurgany_povrezhdennye', 'kurgany_tselye'}
F = json.load(open(oof, encoding='utf-8'))['features']
drop = collections.Counter(); kept = 0
with open(out, 'w', encoding='utf-8') as o, open(st, 'rb') as fh:
    o.write('{"type": "FeatureCollection", "features": ['); first = True
    for f in ijson.items(fh, 'features.item', use_float=True):
        p = f.get('properties') or {}
        if p.get('markup_type') == 'sp_or' and p.get('class_name') in KUR:
            drop[(p.get('threshold_group'), p.get('class_name'))] += 1; continue
        o.write(('' if first else ', ') + json.dumps(f, ensure_ascii=False)); first = False; kept += 1
    for f in F:
        p = f['properties']; assert p['threshold_group'] == 'sp_det2' and p['markup_type'] == 'sp_or' and p['class_name'] in KUR, p
        o.write(', ' + json.dumps({'type': 'Feature', 'geometry': f['geometry'], 'properties': p}, ensure_ascii=False))
    o.write(']}')
print(json.dumps(dict(kept=kept, dropped=sum(drop.values()), added=len(F), drop={f'{g}|{c}': n for (g, c), n in sorted(drop.items(), key=str)}), ensure_ascii=False))
