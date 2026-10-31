"""Сводка geo_val*.json и парные сравнения вариантов (бутстреп по участкам Гнёздово + семьи остальных).
pixgbm готов не для всех участков: для честного сравнения участки без READY берутся из прогона стека (склейка по участкам,
счётчики участков независимы)."""
import sys, json, pickle, copy
sys.path.insert(0, '/workspace/rerank')
import numpy as np
from common import *
from expds.metric import score_counts, paired_bootstrap_counts
from expds.naming import site_family
B = pickle.load(open(W/'work/base_honest.pkl', 'rb'))
def rows(p): return {r['name']: r for r in json.load(open(p))['rows']}
ST = rows('geo_val.json'); PS = rows('geo_val_pixS.json'); PU = rows('geo_val_pixU.json')
READY = set(json.load(open('geo_val_pixS.json'))['pix']['ready'])
def counts(row, stack_row=None):
    c = copy.deepcopy(B['counts']); op = c['object_poly']
    for r, v in row['geo_regions'].items():
        if stack_row is not None and r not in READY: v = stack_row['geo_regions'][r]
        op[r] = list(v) if v else [0, 0, op.get(r, [0, 0, 0])[2]]
    return c
GEOR = set(ST['A_drop']['geo_regions'])
site_of = lambda r: r if r in GEOR else site_family(r)
def cmp(a, b, name, out):
    s = paired_bootstrap_counts(a, b, site_of, 2000, seed=1)
    d = score_counts(a)[0] - score_counts(b)[0]
    out.append(dict(cmp=name, delta=round(d, 6), p5=round(s['p5'], 6), p95=round(s['p95'], 6), p_gt0=round(s['p_gt0'], 3)))
    print(f"{name:60s} Δ {d:+.6f}  site-boot90% [{s['p5']:+.6f}, {s['p95']:+.6f}]  P(>0) {s['p_gt0']:.3f}")
out = []
drop = counts(ST['A_drop']); keep = counts(ST['keep_all'])
for k in ['LOSO_orig_a0', 'LOSO_hull_a0.0', 'LOSO_pole0.05_a0.0', 'LOSO_nested(var,amin,τ)', 'fixed_cfg_etgr0.096635_terfl=keep']:
    cmp(counts(ST[k]), keep, f'stack {k} − keep_all', out)
cmp(counts(ST['LOSO_pole0.05_a0.0']), counts(ST['LOSO_orig_a0']), 'stack pole0.05 − orig (LOSO)', out)
cmp(counts(ST['LOSO_hull_a0.0']), counts(ST['LOSO_orig_a0']), 'stack hull − orig (LOSO)', out)
for tag, P in (('S', PS), ('U', PU)):
    for k in ['LOSO_orig_a0', 'LOSO_pole0.05_a0.0', 'LOSO_nested(var,amin,τ)']:
        a = counts(P[k], ST[k])
        print(f'pix{tag} {k}: final {score_counts(a)[0]:.6f}  opoly', {x: round(y, 4) for x, y in score_counts(a)[1]['object_poly'].items()})
        cmp(a, drop, f'pix{tag} {k} − A_drop', out)
        cmp(a, counts(ST[k]), f'pix{tag} {k} − stack {k}', out)
    a = counts(P['keep_all'], ST['keep_all']); cmp(a, keep, f'pix{tag} keep_all − stack keep_all', out)
json.dump(out, open('compare.json', 'w'), ensure_ascii=False, indent=1)
