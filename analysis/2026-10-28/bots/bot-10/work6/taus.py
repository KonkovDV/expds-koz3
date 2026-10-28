"""Пороги (группа, класс) каждого фолда для раскладок k10/lofo/seed0-4 по кривым базы A (work/curves) и kur 0,5 м (bot-7 curves_kur)."""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo'); sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from expds import metric as M
from expds.naming import site_family
from honest_bootstrap import load_curves, regions_of, random_family_folds
name, src = sys.argv[1], Path(sys.argv[2])
cv = load_curves(src); regions = regions_of(cv)
lay = {'k10': M.family_folds(regions, site_family, 10), 'lofo': M.family_folds(regions, site_family, None)}
for s in range(5): lay[f'seed{s}'] = random_family_folds(regions, 10, s)
out = {'regions': sorted(regions), 'layouts': {}}
for L, folds in lay.items():
    t = time.time(); every = set().union(*folds); fl = []
    for held in folds:
        res = M.sweep_groups({c: M.curve_where(v, every - held) for c, v in cv.items()})
        taus = {}
        for c, v in res.items():
            for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
        fl.append({'held': sorted(held), 'taus': taus})
    out['layouts'][L] = fl; print(name, L, len(folds), round(time.time() - t), 's', flush=True)
json.dump(out, open(f'taus_{name}.json', 'w'), ensure_ascii=False, default=float)
