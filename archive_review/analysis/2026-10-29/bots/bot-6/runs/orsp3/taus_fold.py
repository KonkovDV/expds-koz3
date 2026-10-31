"""bot-6 ORSP v3: пороги sp_focal/sp_focal2 по фолдам (обучающие регионы фолда) для кривых слияния; медиана для сдачи.
Вход: кривые слияния (аргумент 1, по умолчанию свежие curves_orsp3_g), выход taus_fold.json."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from expds import metric as M
from expds.naming import site_family
from honest_bootstrap import load_curves, regions_of, random_family_folds
src = Path(sys.argv[1]); cv = load_curves(src); regions = regions_of(cv)
lay = {'k10': M.family_folds(regions, site_family, 10), 'lofo': M.family_folds(regions, site_family, None)}
for s in range(5): lay[f'seed{s}'] = random_family_folds(regions, 10, s)
K = ('kurgany_povrezhdennye', 'kurgany_tselye'); G = ('sp_focal', 'sp_focal2')
cvk = {c: cv[c] for c in K}
def taus(regs):
    res = M.sweep_groups({c: M.curve_where(v, regs) for c, v in cvk.items()})
    return {g: {c: float(res[c]['groups'][g]['threshold']) for c in K if g in res[c]['groups']} for g in G}
out = {'src': str(src), 'all': taus(regions), 'layouts': {}}
for ln, folds in lay.items():
    out['layouts'][ln] = [taus(regions - held) for held in folds]
med = {}
for g in G:
    for c in K:
        v = [t[g][c] for t in out['layouts']['k10'] if c in t[g]]
        med.setdefault(g, {})[c] = dict(median=float(np.median(v)), min=float(min(v)), max=float(max(v)), n=len(v), values=v)
out['k10_median'] = med
json.dump(out, open('taus_fold.json', 'w'), ensure_ascii=False, indent=1)
print('all', out['all']); print(json.dumps(med, indent=1))
