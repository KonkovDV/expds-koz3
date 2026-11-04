"""P-W2-11 M1/M3: сверка CPU-карт (p11/maps/<tag>) с Kaggle-картами (w2/maps) + Li F1 по участкам фолда 4 (box-декод vs OOF s1 f4 ядра).
usage: m1_compare.py   (после run_m1.sh/run_m3.sh и seed_merge_cnx.py → p11/dec_box_s1f4.geojson)"""
import sys, json, glob, numpy as np
sys.path.insert(0, '/workspace/koz3/w2/vendor'); import compute_metrics as CM
W = '/workspace/koz3/w2'; P = f'{W}/p11'
KM = {'s1': (f'{W}/maps/s1', 's1'), 'm3_s1': (f'{W}/maps/s1', 's1'), 'm3_s0': (f'{W}/maps/s0', 's0'), 'm3_s2': (f'{W}/maps/s2', 's2')}
def kag(root, f, site):
    g = glob.glob(f'{root}/**/maps_cnx_c3-sc_f{f}_{site}.npz', recursive=True); return g[0] if g else None
mapcmp = []
for tag, (root, sd) in KM.items():
    for p in sorted(glob.glob(f'{P}/maps/{tag}/maps_cnx_*.npz')):
        a = np.load(p); m = json.loads(str(a['meta'])); kp = kag(root, m['fold'], m['region'])
        if not kp: mapcmp.append(dict(tag=tag, site=m['region'], err='no kaggle map')); continue
        b = np.load(kp); r = dict(tag=tag, seed=sd, fold=m['fold'], site=m['region'], kaggle=kp.replace(W + '/', ''))
        for k in ('mask', 'hm', 'wh', 'off'):
            x, y = a[k].astype(np.float32), b[k].astype(np.float32)
            if k == 'mask': x, y = x / 255, y / 255
            r[f'mad_{k}'] = float(np.abs(x - y).mean()); r[f'max_{k}'] = float(np.abs(x - y).max())
        r['valid_eq'] = bool((a['valid'] == b['valid']).all()); mapcmp.append(r)
GT = {}
for x in json.load(open(f'{W}/oof/s1/koz3-cnx-li-c3sc-s1f4/gt_li_cnx_c3-sc_f4.geojson', encoding='utf-8'))['features']:
    GT.setdefault(x['properties'].get('region_name'), []).append(x)
def feats(p): return json.load(open(p, encoding='utf-8'))['features']
ARMS = {'oof_kernel': f'{W}/oof/s1/koz3-cnx-li-c3sc-s1f4/oof_cnx_c3-sc_f4.geojson', 'dec_kaggle_maps': f'{P}/dec_kaggle_s1f4.geojson', 'dec_box': f'{P}/dec_box_s1f4.geojson'}
PR = {a: feats(p) for a, p in ARMS.items()}
fc = lambda F: {'type': 'FeatureCollection', 'features': F}; calc = CM.F1Calculator(); TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
def f1(a, r, t): return calc.calculate_f1(fc([x for x in PR[a] if x['properties']['region_name'] == r and x['properties']['confidence'] >= t]), fc(GT.get(r, [])))['final_score']
sites = sorted({x['properties']['region_name'] for x in PR['oof_kernel']} | set(GT))
per = {}
for r in sites:
    d = dict(n_gt=len(GT.get(r, [])))
    for a in PR:
        d[a] = dict(n=sum(x['properties']['region_name'] == r for x in PR[a]))
        if d['n_gt']: d[a].update(f1_t03=round(f1(a, r, 0.3), 5), f1_oracle=round(max(f1(a, r, t) for t in TH), 5))
    per[r] = d
pool = {a: round(calc.calculate_f1(fc([x for x in PR[a] if x['properties']['confidence'] >= 0.3]), fc([g for v in GT.values() for g in v]))['final_score'], 5) for a in PR}
out = dict(map_compare=mapcmp, per_site_f4=per, pool_f4_t03=pool, n_total={a: len(v) for a, v in PR.items()})
json.dump(out, open(f'{P}/result_P-W2-11_m1.json', 'w'), ensure_ascii=False, indent=1)
for r in mapcmp: print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != 'kaggle'})
for r, d in per.items(): print(r, d)
print('POOL', pool, out['n_total'])
