"""[bot-7: копия bot-4/work/dem/dem_eval.py; кандидаты из bot-4/work/dem/out + bot-7/work/dem/out; кэш bot-7; +по классам, LOFO, 5 раскладок]
Оценка грубого прохода gor_coarse_dem: база A (стек stack_arch, table.pkl) против A + кандидаты dem/out/*.geojson.
Честно (family_folds 10, как EXPERIMENTS.md) и all-data; глобально и на покрытом подмножестве участков."""
import sys, json, glob, pickle, collections, time
import numpy as np, shapely
from pathlib import Path
ROOT = Path('/workspace/expds-koz3'); sys.path.insert(0, str(ROOT))
from expds import metric as M
M._SHM_CACHE_DIR = Path('/workspace/koz3_shared/bot-7/work/utm_cache')
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, sweep_groups, paired_bootstrap_counts, _f1, CLASS_WEIGHTS
from expds.naming import site_family

HERE = Path(__file__).parent
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; GOR = 'gorodishcha'; kg = CL.index(GOR)
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
G = shapely.from_wkb(pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))[GOR])
# new candidates
feats = []; covered = []
SRC = [Path('/workspace/koz3_shared/bot-4/work/dem/out'), HERE / 'out']
ONLY = set(sys.argv[1].split(',')) if len(sys.argv) > 1 and sys.argv[1] != 'all' else None
TAG = sys.argv[2] if len(sys.argv) > 2 else 'all'
for p in sorted(q for d in SRC for q in glob.glob(str(d / '*.geojson'))):
    if ONLY is not None and Path(p).parent.parts[-4] not in ONLY: continue
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
def random_family_folds(regions, k, seed):
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    names = sorted(fams); rng = np.random.default_rng(seed); lab = rng.permutation(len(names)) % k
    return [set().union(*(fams[n] for n, l in zip(names, lab) if l == i)) for i in range(k)]
lofo_folds = family_folds(regions, site_family, None)
seed_folds = [random_family_folds(regions, 10, s) for s in range(5)]
covset = set(covered); gt_cov = sum(1 for r in T['greg'][GOR] if r in covset)
def run(scale):
    cur = dict(others); cur[GOR] = gor_curve(scale)
    cnt, info = honest_counts(cur, folds)
    res = sweep_groups(cur); taus = {}
    for c, v in res.items():
        for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
    ins = M.score_group_thresholds(cur, taus)[0]
    sg = score_counts(cnt)[1][GOR]; sc = score_counts(cnt, covset)
    pc = {c: {k: x[k] for k in ('f1', 'tp', 'fp', 'n_gt') if k in x} for c, x in score_counts(cnt)[1].items()}
    lofo = score_counts(honest_counts(cur, lofo_folds)[0])[0]
    seeds = [score_counts(honest_counts(cur, f)[0])[0] for f in seed_folds]
    return dict(honest=score_counts(cnt)[0], per_class=pc, lofo=lofo, seeds=seeds, allData=ins, gor_honest=sg, honest_cov=sc[0], gor_honest_cov=sc[1][GOR],
                gor_alldata_f1=res[GOR]['f1'], gor_groups=res[GOR]['groups'], info=info), cnt
out = {'covered_regions': covered, 'n_regions': len(covered), 'gor_gt_covered': gt_cov, 'gor_gt_total': len(G),
       'n_new': len(new), 'new_with_pair': len(newpairs)}
base, cb = run(None); out['base'] = base
for sc in [1.0, 0.4]:
    r, cn = run(sc)
    r['delta_honest'] = r['honest'] - base['honest']; r['delta_all'] = r['allData'] - base['allData']
    r['delta_honest_cov'] = r['honest_cov'] - base['honest_cov']
    r['delta_lofo'] = r['lofo'] - base['lofo']; r['delta_seeds'] = [a - b for a, b in zip(r['seeds'], base['seeds'])]
    r['delta_per_class_weighted'] = {c: CLASS_WEIGHTS[c] / 100 * (r['per_class'][c]['f1'] - base['per_class'][c]['f1']) for c in r['per_class'] if c in base['per_class']}
    r['delta_gor_f1'] = r['per_class'][GOR]['f1'] - base['per_class'][GOR]['f1']
    try:
        bs = paired_bootstrap_counts(cn, cb, site_family, n=2000, seed=0)
        r['bootstrap'] = bs
    except Exception as e:
        r['bootstrap'] = f'err {e!r}'
    out[f'scale_{sc}'] = r
    print(sc, {k: v for k, v in r.items() if k.startswith('delta') or k in ('honest', 'allData')}, flush=True)
print('base', base['honest'], base['allData'], base['honest_cov'])
json.dump(out, open(HERE / f'res_dem_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
