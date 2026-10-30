"""W2: стек A + один OOF cnx (одиночный сид) как отдельная threshold_group li_cnx (аналог V6c у W5, без слияния сидов).
usage: stack_add_cnx.py <stack_arch.geojson> <oof_cnx.geojson> <out.geojson> <fid_prefix>
Байтовое дописывание в конец features (как cnx_feats.py W5); conf без изменений; threshold_group = li_cnx."""
import json, sys
st, oof, out, pre = sys.argv[1:5]
F = json.load(open(oof, encoding='utf-8'))['features']
with open(st, 'rb') as fh:
    fh.seek(0, 2); n = fh.tell(); fh.seek(n - 64); tail = fh.read()
k = tail.rstrip().rfind(b']}'); assert k > 0; cut = n - 64 + k
with open(out, 'wb') as o, open(st, 'rb') as fh:
    left = cut
    while left > 0:
        b = fh.read(min(1 << 24, left)); o.write(b); left -= len(b)
    for i, f in enumerate(F):
        p = dict(f['properties']); assert p['threshold_group'] == 'li_cnx'; p['fid'] = f'{pre}_{i}'
        o.write(b', ' + json.dumps({'type': 'Feature', 'geometry': f['geometry'], 'properties': p}, ensure_ascii=False).encode())
    o.write(b']}')
print('appended', len(F), '->', out)
