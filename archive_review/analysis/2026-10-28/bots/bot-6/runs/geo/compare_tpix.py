"""Сводка pixgbm по одному уровню t_pix (geo_val_pix{S,U}[M]_t{0.3,0.5,0.7}.json) против стека (geo_val.json).
Все участки READY, склейки со стеком нет. Парный бутстреп по участкам (Гнёздово — по участку, прочие — по семье), 2000."""
import sys, json, pickle, copy
sys.path.insert(0, '/workspace/rerank')
from common import *
from expds.metric import score_counts, paired_bootstrap_counts
from expds.naming import site_family
B = pickle.load(open(W/'work/base_honest.pkl', 'rb'))
def load(p):
    d = json.load(open(p)); return {r['name']: r for r in d['rows']}
def counts(row):
    c = copy.deepcopy(B['counts'])
    if 'op_counts' in row:  # весь object_poly прогона (в режиме M меняются и участки maggr 200–211)
        c['object_poly'] = {r: list(v) for r, v in row['op_counts'].items()}; return c
    op = c['object_poly']
    for r, v in row['geo_regions'].items():
        op[r] = list(v) if v else [0, 0, op.get(r, [0, 0, 0])[2]]
    return c
ST = load('geo_val.json')
GEOR = set(ST['A_drop']['geo_regions'])
site_of = lambda r: r if r[:3] in {'208','209','210','211','212','213','214','215','216','217'} else site_family(r)
KEYS = ['A_drop', 'keep_all', 'LOSO_orig_a0', 'LOSO_hull_a0.0', 'LOSO_nested(var,amin,τ)']
out = {'stack': {k: ST[k]['honest10'] for k in KEYS}, 'runs': []}
def cmp(a, b):
    s = paired_bootstrap_counts(a, b, site_of, 2000, seed=1)
    return dict(delta=round(score_counts(a)[0] - score_counts(b)[0], 6), p5=round(s['p5'], 6), p95=round(s['p95'], 6), p_gt0=round(s['p_gt0'], 3))
stc = {k: counts(ST[k]) for k in KEYS}
for k in KEYS:
    assert abs(score_counts(stc[k])[0] - ST[k]['honest10']) < 2e-6, (k, score_counts(stc[k])[0], ST[k]['honest10'])
for t in ('0.3', '0.5', '0.7'):
    for m in ('S', 'U', 'SM', 'UM'):
        f = f'geo_val_pix{m}_t{t}.json'
        try: P = load(f)
        except FileNotFoundError: print('нет', f); continue
        rec = {'file': f, 'tpix': t, 'mode': m}
        for k in KEYS:
            c = counts(P[k]); sc = score_counts(c)[0]
            rec[k] = {'honest10': P[k]['honest10'], 'recount': round(sc, 6), 'opoly_f1': round(score_counts(c)[1]['object_poly']['f1'], 4)}
        n = counts(P['LOSO_nested(var,amin,τ)'])
        rec['nested_vs_stack_nested'] = cmp(n, stc['LOSO_nested(var,amin,τ)'])
        rec['nested_vs_stack_keep_all'] = cmp(n, stc['keep_all'])
        rec['keep_all_vs_stack_keep_all'] = cmp(counts(P['keep_all']), stc['keep_all'])
        out['runs'].append(rec)
        print(f"t{t} {m:2s} nested {rec['LOSO_nested(var,amin,τ)']['honest10']:.6f} (пересчёт {rec['LOSO_nested(var,amin,τ)']['recount']:.6f}) "
              f"keep {rec['keep_all']['honest10']:.6f} hull0 {rec['LOSO_hull_a0.0']['honest10']:.6f} | "
              f"nested−stack_nested {rec['nested_vs_stack_nested']} | nested−stack_keep {rec['nested_vs_stack_keep_all']}")
json.dump(out, open('compare_tpix.json', 'w'), ensure_ascii=False, indent=1)
