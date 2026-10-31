"""bot-10 work11 задача 4: распределение уверенностей LOFO-OOF PIXGBM (bot-3/pixgbm/pix_lofo.geojson) по (группа, класс):
квантили, число и доля выше порогов thr_pix_full.json; N(>=t) на сетке t — для квантильного согласования со сдачей all-модели."""
import json, collections
import numpy as np
d = json.load(open('/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson', encoding='utf-8'))
T = json.load(open('/workspace/koz3_shared/bot-3/pixgbm/pc/thr_pix_full.json'))
C = collections.defaultdict(list); R = collections.defaultdict(set); keys = collections.Counter()
for f in d['features']:
    p = f['properties']; g = p.get('threshold_group') or p.get('markup_type'); keys[tuple(sorted(p))] += 1
    C[(g, p['class_name'])].append(float(p['confidence'])); R[(g, p['class_name'])].add(p.get('region_name'))
print('property keys', keys.most_common(2))
out = {}
for (g, c), v in sorted(C.items()):
    v = np.asarray(v); tau = T.get(f'S|global|{c}', {}).get(g)
    q = {str(x): float(np.quantile(v, x)) for x in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
    grid = {f'{t:.2f}': int((v >= t).sum()) for t in np.arange(0.5, 1.0001, 0.02)}
    r = {'n': int(len(v)), 'regions': len(R[(g, c)]), 'quantiles': q, 'tau': tau,
         'n_ge_tau': int((v >= tau).sum()) if tau is not None else None, 'frac_ge_tau': float((v >= tau).mean()) if tau is not None else None, 'n_ge_t': grid}
    out[f'{g}|{c}'] = r
    print(g, c, 'n', r['n'], 'regions', r['regions'], 'tau', tau, 'n>=tau', r['n_ge_tau'], 'frac', None if tau is None else round(r['frac_ge_tau'], 4),
          'q', {k: round(x, 4) for k, x in q.items()})
json.dump(out, open('pix_lofo_counts.json', 'w'), ensure_ascii=False, indent=1)
