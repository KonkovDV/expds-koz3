"""Диагностика полноты без лидара (bot-5). Стек /workspace/rerank/stack_arch.geojson через table.pkl
(официальные пары _match_quality). Разделяет: потолок кандидатов, пороги/ранжирование, геометрию/цель."""
import sys, json, pickle, collections, time
import numpy as np, shapely
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
ROOT = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(ROOT))
from expds import metric as M
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, sweep_groups, curve_where
from expds.naming import site_family
M._SHM_CACHE_DIR = Path('/workspace/koz3_shared/bot-5/tmp/shm_off/x')
OUT = Path('/workspace/koz3_shared/bot-5/analysis/nolidar'); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; cls, conf, reg, grp = T['cls'], T['conf'], T['reg'], T['grp']
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
gmk = {c: [] for c in CL}
for f in gt['features']:
    p = f.get('properties') or {}; c = mod.normalize_class_name(p.get('class_name', 'unknown'))
    if c in gmk and f.get('geometry'): gmk[c].append(str(p.get('markup_type')))
del gt
for c in CL: assert len(gmk[c]) == T['n_gt'][c]
gwkb = pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))
GG = {c: shapely.from_wkb(gwkb[c]) for c in CL}
boxes = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
FOC = KUR + ['gorodishcha', 'arkhitektury', 'fortifikatsii']

def greedy(c, cand, pc=None):
    P, J, S = T['pairs'][(pc or c, c)]
    by = collections.defaultdict(list)
    for a, b, s in zip(P, J, S): by[a].append((b, s))
    matched = {}; used = set()
    for i in cand:
        lst = by.get(i)
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in sorted(lst):
            if j in used: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); matched[i] = bj
    return matched
order = {}; curves = {}
for k, c in enumerate(CL):
    idx = np.flatnonzero(cls == k); o = idx[np.argsort(-conf[idx], kind='stable')]; order[c] = o
    m = greedy(c, o); hit = np.fromiter((i in m for i in o), bool, len(o))
    curves[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                           gt_regions=collections.Counter(T['greg'][c]), pred_group=list(grp[o]))
regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10)
counts, info = honest_counts(curves, folds); honest_A = score_counts(counts)[0]
print('honest A', honest_A, flush=True)
every = set().union(*folds)
kept_h = np.zeros(len(cls), bool)
for held in folds:
    res = sweep_groups({c: curve_where(cv, every - held) for c, cv in curves.items()})
    for c in CL:
        taus = {g: gv['threshold'] for g, gv in res.get(c, {'groups': {}})['groups'].items()}
        o = order[c]; inh = np.isin(reg[o], list(held)); tau = np.array([taus.get(g, np.inf) for g in grp[o]])
        kept_h[o[inh & (conf[o] >= tau)]] = True
res_all = sweep_groups(curves); kept_i = np.zeros(len(cls), bool)
for c in CL:
    taus = {g: gv['threshold'] for g, gv in res_all[c]['groups'].items()}; o = order[c]
    tau = np.array([taus.get(g, np.inf) for g in grp[o]]); kept_i[o[conf[o] >= tau]] = True
print('kept', kept_h.sum(), kept_i.sum(), time.time() - t0, flush=True)

def oracle(c, mask_rows=None):
    P, J, S = T['pairs'][(c, c)]
    if mask_rows is not None:
        s = mask_rows[P]; P, J = P[s], J[s]
    if not len(P): return np.zeros(T['n_gt'][c], bool)
    up, pi = np.unique(P, return_inverse=True)
    A = csr_matrix((np.ones(len(P)), (J, pi)), shape=(T['n_gt'][c], len(up)))
    m = maximum_bipartite_matching(A, perm_type='column'); return m >= 0
def q(a, ps=(10, 50, 90)): return [round(float(x), 1) for x in np.percentile(a, ps)] if len(a) else []

R = {'honest_A': honest_A, 'cells': {}, 'groups': {}, 'geometry': {}}
for c in FOC:
    k = CL.index(c); ng = T['n_gt'][c]; gm = np.array(gmk[c]); gr = np.array(T['greg'][c])
    nol = ~np.isin(gr, list(boxes))
    P, J, S = T['pairs'][(c, c)]
    mh = greedy(c, order[c][kept_h[order[c]]]); tp_h = np.zeros(ng, bool); tp_h[list(mh.values())] = True
    mi = greedy(c, order[c][kept_i[order[c]]]); tp_i = np.zeros(ng, bool); tp_i[list(mi.values())] = True
    cov = np.zeros(ng, bool); cov[J] = True
    orc = oracle(c)
    # кандидаты другого класса курганов, которые сопоставимы с этим GT (путаница классов)
    other = np.zeros(ng, bool)
    if c in KUR:
        oc = KUR[1 - KUR.index(c)]; P2, J2, _ = T['pairs'][(oc, c)]; other[J2] = True
    # покрытие по группам
    gcov = collections.defaultdict(lambda: np.zeros(ng, bool))
    for a, b in zip(P, J): gcov[grp[a]][b] = True
    # ближайший кандидат того же класса в том же участке
    idx = np.flatnonzero(cls == k); cx, cy = T['cx'][idx], T['cy'][idx]; rr = reg[idx]
    G = GG[c]; gx, gy = T['gx'][c], T['gy'][c]
    near_in = np.zeros(ng, bool); dmin = np.full(ng, np.inf)
    byreg = collections.defaultdict(list)
    for i, r in enumerate(rr): byreg[r].append(i)
    byreg = {r: np.array(v) for r, v in byreg.items()}
    for j in range(ng):
        ii = byreg.get(gr[j])
        if ii is None: continue
        d = np.hypot(cx[ii] - gx[j], cy[ii] - gy[j]); dmin[j] = d.min()
        cand = ii[d < 60]
        if len(cand) and not cov[j]:
            near_in[j] = bool(shapely.contains_xy(G[j], cx[cand], cy[cand]).any())
    for mk in sorted(set(gm)):
        for lid in (False, True):
            s = (gm == mk) & (nol != lid)
            if not s.any(): continue
            unc = s & ~cov
            cell = dict(n_gt=int(s.sum()), honest_tp=int((tp_h & s).sum()), insample_tp=int((tp_i & s).sum()),
                        covered=int((cov & s).sum()), oracle_tp=int((orc & s).sum()),
                        other_kurgan_class_only=int((other & unc).sum()),
                        uncovered=int(unc.sum()),
                        unc_centroid_inside_gt=int((near_in & unc).sum()),
                        unc_nearest_le5m=int((unc & (dmin <= 5)).sum()), unc_5_20m=int((unc & (dmin > 5) & (dmin <= 20)).sum()),
                        unc_gt20m=int((unc & (dmin > 20) & np.isfinite(dmin)).sum()), unc_no_cand_in_region=int((unc & ~np.isfinite(dmin)).sum()),
                        cov_by_group={g: int((v & s).sum()) for g, v in sorted(gcov.items()) if (v & s).any()},
                        gt_area_p10_50_90=q(shapely.area(G[s])))
            R['cells'][f'{c}|{mk}|{"lidar" if lid else "nolidar"}'] = cell
    # группы: кандидаты, оставлено честно, TP честно/in-sample, в участках без лидара
    for g in sorted(set(grp[idx])):
        rows = idx[grp[idx] == g]; nl = ~np.isin(reg[rows], list(boxes))
        rows = rows[nl]
        if not len(rows): continue
        R['groups'][f'{c}|{g}'] = dict(cands_nolidar=int(len(rows)), kept_honest=int(kept_h[rows].sum()),
            tp_honest=int(sum(1 for i in rows if i in mh)), kept_insample=int(kept_i[rows].sum()),
            tp_insample=int(sum(1 for i in rows if i in mi)),
            area_p10_50_90=q(T['area'][rows]))
    print(c, time.time() - t0, flush=True)

# геометрия/цель для курганов: склейка масок соседей и подавление пиков
for mk in ('sp_or', 'ae', 'or', 'li'):
    geoms = []; meta = []
    for c in KUR:
        s = np.array(gmk[c]) == mk
        geoms += list(GG[c][s]); meta += [(c, r) for r in np.array(T['greg'][c])[s]]
    if not geoms: continue
    geoms = np.array(geoms, dtype=object); regs_ = np.array([m[1] for m in meta])
    tree = shapely.STRtree(geoms)
    out = {'n': len(geoms)}
    for tol in (0.5, 1.0, 2.0):
        a, b = tree.query(geoms, predicate='dwithin', distance=tol); s = (a != b) & (regs_[a] == regs_[b])
        out[f'touch_le_{tol}m'] = int(len(np.unique(a[s])))
    cen = shapely.centroid(geoms); X, Y = shapely.get_x(cen), shapely.get_y(cen)
    for md in (4, 12, 24):
        sup = 0
        for r in np.unique(regs_):
            ii = np.flatnonzero(regs_ == r); ii = ii[np.argsort(-shapely.area(geoms[ii]))]
            keep = []
            for i in ii:
                if all((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 >= md * md for j in keep): keep.append(i)
            sup += len(ii) - len(keep)
        out[f'suppressed_by_min_dist_{md}m'] = sup
    # идеальный центр: круг радиуса r в представительной точке GT — проходит ли официальная проверка
    from shapely.geometry import Point
    rp = shapely.point_on_surface(geoms)
    for rad in (2, 4, 6, 10):
        ok = sum(calc._match_quality(Point(p.x, p.y).buffer(rad, 32), g)[0] for p, g in zip(rp, geoms))
        out[f'perfect_center_circle_r{rad}_ok'] = int(ok)
    out['area_p10_50_90'] = q(shapely.area(geoms))
    R['geometry'][mk] = out
print('geometry', time.time() - t0, flush=True)

# городища: окно модели sp_or (tile 128 px × 2 м = 256 м) против размера GT
c = 'gorodishcha'; G = GG[c]; b = shapely.bounds(G); side = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
gm = np.array(gmk[c])
R['geometry']['gorodishcha'] = {mk: dict(n=int((gm == mk).sum()), side_p10_50_90=q(side[gm == mk]),
                                        side_gt_128m=int(((gm == mk) & (side > 128)).sum()),
                                        side_gt_256m=int(((gm == mk) & (side > 256)).sum()))
                                for mk in sorted(set(gm))}
json.dump(R, open(OUT / 'nolidar_diag.json', 'w'), ensure_ascii=False, indent=1)
print('done', time.time() - t0)
