"""P-W2-10: Li-only F1 по участкам (compute_metrics вендора). Порог conf: единый, выбран по K (Kaggle s012avg) на пуле 4 участков; плюс oracle-порог на участок.
usage: liscore.py out.json K=<geojson> C=<geojson> T=<geojson>"""
import sys, json
sys.path.insert(0, '/workspace/koz3/w2/vendor'); import compute_metrics as CM
W = '/workspace/koz3/w2/oof/s2'
REG = {'015_Демидовка': 2, '021_Новенькое': 3, '186_Семибратнее_1': 3, '190_Муравлево': 4}
GT = {}
for f, p in ((2, f'{W}/koz3-w2-cnx-c3sc-s2f23/gt_li_cnx_c3-sc_f2.geojson'), (3, f'{W}/koz3-w2-cnx-c3sc-s2f23/gt_li_cnx_c3-sc_f3.geojson'), (4, f'{W}/koz3-w2-cnx-c3sc-s2f4/gt_li_cnx_c3-sc_f4.geojson')):
    for x in json.load(open(p, encoding='utf-8'))['features']: GT.setdefault(x['properties'].get('region_name'), []).append(x)
arms = dict(a.split('=', 1) for a in sys.argv[2:])
P = {k: [x for x in json.load(open(v, encoding='utf-8'))['features'] if x['properties'].get('region_name') in REG] for k, v in arms.items()}
fc = lambda F: {'type': 'FeatureCollection', 'features': F}
calc = CM.F1Calculator()
def f1(arm, regs, thr): return calc.calculate_f1(fc([x for x in P[arm] if x['properties']['region_name'] in regs and x['properties']['confidence'] >= thr]), fc([g for r in regs for g in GT.get(r, [])]))['final_score']
TH = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
pool = {t: f1('K', list(REG), t) for t in TH}; tg = max(pool, key=pool.get)
out = dict(thr_global=tg, pool_K=pool, n_gt={r: len(GT.get(r, [])) for r in REG}, per_site={})
for r in REG:
    d = {}
    for a in P:
        d[a] = dict(f1_tg=round(f1(a, [r], tg), 5), f1_oracle=round(max(f1(a, [r], t) for t in TH), 5), n_pred=sum(x['properties']['region_name'] == r for x in P[a]))
    out['per_site'][r] = d
out['pool'] = {a: round(f1(a, list(REG), tg), 5) for a in P}
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False))
