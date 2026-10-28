"""audit01: честные П1–П5 для maps=all / dedupe_maps=false / band auto (maggr, 201–211; 200 и 212 — из стека)."""
import sys, json, collections
from evlib import *
RUN = {f'{n}' for n in ['201_Семибратнее_Городище_1','202_Семибратнее_Городище_2','203_Семибратнее_Городище_3','204_Семибратнее_Городище_4',
       '205_Семибратнее_Городище_5','206_Семибратнее_Городище_6','207_Семибратнее_Городище_7','208_Гнёздово_Пойма',
       '209_Гнёздово_Терраса_1','210_Гнёздово_Терраса_2','211_Гнёздово_Терраса_3']}
variants = sys.argv[1:] or ['A1', 'A2', 'A3']
def build(name):
    fp = file_preds(f'a01/{name}.geojson')
    return {c: swap(c, RUN, fp.get(c, [])) for c in GEO}, fp
PB, fpB = build('B')
out = {}
for scope in ['global', 'site']:
    cB = curves_of(PB, scope)
    for name in variants:
        PV, fpV = build(name)
        cV = curves_of(PV, scope)
        for unseen in ['drop', 'keep']:
            r = criteria(cV, cB, unseen)
            key = f'{name}|{scope}|{unseen}'
            # по участкам 208–211, finds
            a, _ = counts_with(cV, unseen=unseen); b, _ = counts_with(cB, unseen=unseen)
            r['finds_by_region'] = {reg: {'var': a['finds_points'].get(reg), 'base': b['finds_points'].get(reg)}
                                    for reg in sorted(RUN) if reg[:3] in ('208', '209', '210', '211')}
            out[key] = r
            print(key, 'base', [round(x, 6) for x in r['base']], 'var', [round(x, 6) for x in r['var']],
                  'd10 %+.6f lofo %+.6f rnd %s P(>0) %.3f' % (r['d_honest10'], r['d_lofo'], [round(x, 5) for x in r['d_rand']], r['boot'].get('p_gt0', 0)),
                  'P1-4', r['P1'], r['P2'], r['P3'], r['P4'], 'belowMDE', r['P5_below_MDE'], flush=True)
            print('   finds', {k: r['per_class_var']['finds_points'][k] for k in ('tp','fp','f1')}, 'vs', {k: r['per_class_base']['finds_points'][k] for k in ('tp','fp','f1')},
                  '| opoly', {k: r['per_class_var']['object_poly'][k] for k in ('tp','fp','f1')}, 'vs', {k: r['per_class_base']['object_poly'][k] for k in ('tp','fp','f1')}, flush=True)
    # дубли: находки разных карт одного участка ближе 2 м (сырой файл варианта)
for name in ['B'] + variants:
    fp = json.load(open(f'a01/{name}.geojson'))
    cnt = collections.Counter((f['properties']['region_name'][:3], f['properties']['class_name'], f['properties'].get('source')) for f in fp['features'])
    out[f'{name}|sources'] = {'|'.join(k): v for k, v in sorted(cnt.items()) if k[0] in ('208','209','210','211')}
json.dump(out, open('a01/eval_' + '_'.join(variants) + '.json', 'w'), ensure_ascii=False, indent=1, default=float)
