"""Оценка грубого прохода gor_coarse_dem: база A (стек stack_arch, table.pkl) против A + кандидаты dem/out/*.geojson.
Честно (family_folds 10, как EXPERIMENTS.md) и all-data; глобально и на покрытом подмножестве участков."""
import sys, json, glob, pickle, collections, time
import numpy as np, shapely
from pathlib import Path
ROOT = Path('/workspace/expds-koz3'); sys.path.insert(0, str(ROOT))
from expds import metric as M
M._SHM_CACHE_DIR = Path('/workspace/koz3_shared/bot-4/work/utm_cache')
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, sweep_groups, paired_bootstrap_counts, _f1, CLASS_WEIGHTS
from expds.naming import site_family
HERE = Path(__file__).parent
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; GOR = 'gorodishcha'; kg = CL.index(GOR)
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
G = shapely.from_wkb(pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))[GOR])
# new candidates
feats = []; covered = []
for p in sorted(glob.glob(str(HERE / 'out' / '*.geojson'))):
    covered.append(Path(p).stem); feats += json.load(open(p))['features']
new = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': feats}, False).get(GOR, [])
assert len(new) == len(feats), (len(new), len(feats))
tree = shapely.STRtree(G)
newpairs = collections.defaultdict(list)
for i, (g, cf, rg, grp) in enumerate(new):
    for j in tree.query(g):
        ok, s = calc._match_quality(g, G[j])
        if ok: newpairs[i].append((int(j), s))
print('new cands', len(new), 'regions', len(covered), 'with pair', len(newpairs), 'GT hit', len({j for v in newpairs.values() for j, _ in v}), flush=True)
base_idx = np.flatnonzero(T['cls'] == kg)
P, J, S = T['pairs'][(GOR, GOR)]
bp = collections.defaultdict(list)
for a, b, s in zip(P, J, S): bp[int(a)].append((int(b), s))
def gor_curve(scale):
    conf = list(T['conf'][base_idx]); reg = list(T['reg'][base_idx]); grp = list(T['grp'][base_idx]); prs = [bp.get(int(i), []) for i in base_idx]
    if scale is not None:
        for i, (g, cf, rg, gp) in enumerate(new):
            conf.append(cf * scale); reg.append(rg); grp.append('gor_coarse_dem'); prs.append(newpairs.get(i, []))
    conf = np.array(conf); o = np.argsort(-conf, kind='stable'); used = set(); hit = np.zeros(len(o), bool)
    for r, i in enumerate(o):
        best, bj = -np.inf, -1
        for j, s in sorted(prs[i]):
            if j in used: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); hit[r] = True
    return ClassCurve(conf=conf[o], hit=hit, pred_region=[reg[i] for i in o], n_gt=len(G),
                      gt_regions=collections.Counter(T['greg'][GOR]), pred_group=[grp[i] for i in o])
# other classes: base curves unchanged (same as gap.py)
def class_curve(k, c):
    idx = np.flatnonzero(T['cls'] == k); o = idx[np.argsort(-T['conf'][idx], kind='stable')]
    P, J, S = T['pairs'][(c, c)]; by = collections.defaultdict(list)
    for a, b, s in zip(P, J, S): by[int(a)].append((int(b), s))
    used = set(); hit = np.zeros(len(o), bool)
    for r, i in enumerate(o):
        best, bj = -np.inf, -1
        for j, s in sorted(by.get(int(i), [])):
            if j in used: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); hit[r] = True
    return ClassCurve(conf=T['conf'][o].astype(float), hit=hit, pred_region=list(T['reg'][o]), n_gt=T['n_gt'][c],
                      gt_regions=collections.Counter(T['greg'][c]), pred_group=list(T['grp'][o]))
others = {c: class_curve(k, c) for k, c in enumerate(CL) if c != GOR}
regions = {r for cv in others.values() for r in cv.pred_region} | {r for cv in others.values() for r in cv.gt_regions} | set(T['reg'][base_idx]) | set(T['greg'][GOR])
folds = family_folds(regions, site_family, 10)
covset = set(covered); gt_cov = sum(1 for r in T['greg'][GOR] if r in covset)
def run(scale):
    cur = dict(others); cur[GOR] = gor_curve(scale)
    cnt, info = honest_counts(cur, folds)
    res = sweep_groups(cur); taus = {}
    for c, v in res.items():
        for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
    ins = M.score_group_thresholds(cur, taus)[0]
    sg = score_counts(cnt)[1][GOR]; sc = score_counts(cnt, covset)
    return dict(honest=score_counts(cnt)[0], allData=ins, gor_honest=sg, honest_cov=sc[0], gor_honest_cov=sc[1][GOR],
                gor_alldata_f1=res[GOR]['f1'], gor_groups=res[GOR]['groups'], info=info), cnt
out = {'covered_regions': covered, 'n_regions': len(covered), 'gor_gt_covered': gt_cov, 'gor_gt_total': len(G),
       'n_new': len(new), 'new_with_pair': len(newpairs)}
base, cb = run(None); out['base'] = base
for sc in [1.0, 0.4]:
    r, cn = run(sc)
    r['delta_honest'] = r['honest'] - base['honest']; r['delta_all'] = r['allData'] - base['allData']
    r['delta_honest_cov'] = r['honest_cov'] - base['honest_cov']
    try:
        bs = paired_bootstrap_counts(cn, cb, site_family, n=2000, seed=0)
        r['bootstrap'] = bs
    except Exception as e:
        r['bootstrap'] = f'err {e!r}'
    out[f'scale_{sc}'] = r
    print(sc, {k: v for k, v in r.items() if k.startswith('delta') or k in ('honest', 'allData')}, flush=True)
print('base', base['honest'], base['allData'], base['honest_cov'])
json.dump(out, open(HERE / 'res_dem.json', 'w'), ensure_ascii=False, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
