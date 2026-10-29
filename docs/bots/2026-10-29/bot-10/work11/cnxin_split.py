"""bot-10 work11 C2b: откуда расхождение S4 (минус) и полного стека (плюс). Счётчики k10 keep полного стека
делятся на участки фолда 4 (COV) и остальные; Δ балла и TP/FP курганов по частям. Только объяснение.
    python work11/cnxin_split.py > work11/cnxin_split.log"""
import sys, json
from pathlib import Path
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
RF = json.load(open('/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/c3sc_f01_out/region_fold.json', encoding='utf-8'))['region_fold']
COV = {r for r, f in RF.items() if f == 4}
KUR = ('kurgany_povrezhdennye', 'kurgany_tselye')
V = {'A': 'work/curves', 'Rin': 'work11/curves_cnxRin_g', 'RCin': 'work11/curves_cnxRCin_g'}
out = {}
for nm, d in V.items():
    cv = load_curves(Path(d))
    regions = regions_of(cv)
    S4reg = {r for r in regions if any(site_family(r) == site_family(c) for c in COV)}
    folds = M.family_folds(regions, site_family, 10)
    c, _ = M.honest_counts(cv, folds, unseen='keep')
    o = {}
    for part, sub in (('all', None), ('cov_fold4', COV & regions), ('s4_units_regions', S4reg), ('rest', regions - S4reg)):
        sc, per = M.score_counts(c, sub) if sub is not None else M.score_counts(c)
        o[part] = {'score': sc, 'n_regions': len(sub) if sub is not None else len(regions),
                   **{k: {x: per[k][x] for x in ('tp', 'fp', 'n_gt', 'f1')} for k in KUR if k in per}}
    out[nm] = o
    print(nm, {p: (round(v['score'], 6), {k: (v[k]['tp'], v[k]['fp']) for k in KUR if k in v}) for p, v in o.items()}, flush=True)
    del cv, c
json.dump(out, open('work11/cnxin_split.json', 'w'), ensure_ascii=False, indent=1)
