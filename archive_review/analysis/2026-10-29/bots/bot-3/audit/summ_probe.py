import json, collections, sys
R = json.load(open('out/sat_probe_ext.json'))
print('n rows', len(R), 'errors', [r['file'] for r in R if 'gt_finds' not in r and 'hp_energy' not in r])
K = ['gt_finds_in_frame','gt_finds_in_satc','gt_finds_in_sat','gt_finds_near_sat05','gt_finds_invalid_S',
     'gt_poly_in_frame','gt_poly_in_satc','gt_poly_in_sat','gt_poly_near_sat05','gt_poly_invalid_S','gt_poly_ge50_sat']
def agg(rows):
    t = collections.Counter()
    for r in rows:
        for k in K: t[k] += r.get(k, 0) or 0
    return t
P = [r for r in R if r['picked']]
print('\n== выбранные карты (как в S), по участку × методу')
print('site mt | finds: n satc sat near0.5 invalidS | poly: n satc sat near invalidS ge50sat | satc_share hp')
for r in P:
    if not (r.get('gt_finds_in_frame') or r.get('gt_poly_in_frame')): 
        print(r['region'][:3], r['mt'], r['file'][:38], 'нет эталона/вне кадра', 'satc_share', r.get('satc_share_of_valid')); continue
    print(r['region'][:3], r['mt'], r['file'][:38], '|', *[r.get(k,0) for k in K[:5]], '|', *[r.get(k,0) for k in K[5:]], '|', r.get('satc_share_of_valid'), r.get('hp_energy'), 'area_sat', r.get('gt_poly_area_share_sat'))
print('\n== итог по методу (выбранные)')
for mt in ('maggr','etgr','terfl'):
    t = agg([r for r in P if r['mt']==mt]); print(mt, dict(t))
t = agg(P); print('ALL', dict(t))
for a,b in [('gt_finds_in_satc','gt_finds_in_frame'),('gt_finds_in_sat','gt_finds_in_frame'),('gt_finds_near_sat05','gt_finds_in_frame'),
            ('gt_poly_in_satc','gt_poly_in_frame'),('gt_poly_in_sat','gt_poly_in_frame'),('gt_poly_ge50_sat','gt_poly_in_frame')]:
    print(f'  {a}/{b} = {t[a]}/{t[b]} = {t[a]/max(t[b],1):.4f}')
print('\n== копии (ключ с >1 файлом)')
by = collections.defaultdict(list)
for r in R: by[(r['region'], r['mt'], r['key'])].append(r)
for k, rs in by.items():
    if len(rs) < 2: continue
    for r in rs:
        print(k[0][:3], r['file'][:40].ljust(40), 'picked', r['picked'], 'hp', r.get('hp_energy'), 'auc', r.get('finds_signal_auc'),
              'gt_f', r.get('gt_finds_in_frame'), 'satF', r.get('gt_finds_in_sat'), 'invS_F', r.get('gt_finds_invalid_S'), 'valid0', r.get('valid0_share'), 'sat', r.get('sat_share'))
