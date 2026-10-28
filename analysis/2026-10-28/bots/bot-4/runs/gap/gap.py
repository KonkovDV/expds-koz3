"""GAP: карта потерь на честной базе A (bot-4). Источник: /workspace/rerank/work/table.pkl
(стек stack_arch.geojson, все 1 386 709 кандидатов, официальные пары _match_quality)."""
import sys, json, pickle, collections, time
import numpy as np, shapely
from pathlib import Path
ROOT = Path('/workspace/expds-koz3'); sys.path.insert(0, str(ROOT))
from expds import metric as M
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, sweep_groups, curve_where, _f1, CLASS_WEIGHTS
from expds.naming import site_family
OUT = Path(__file__).parent
t0 = time.time()
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; W = {c: CLASS_WEIGHTS[c] for c in CL}
TYPE = {'li': 'Li', 'ae': 'Ae', 'or': 'Or', 'sp_or': 'Or', 'maggr': 'Geo', 'etgr': 'Geo', 'terfl': 'Geo'}
# GT markup per G index (same filter/order as _collect: target class, geometry present)
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py'))
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson'))
gmk = {c: [] for c in CL}
for f in gt['features']:
    p = f.get('properties') or {}; c = mod.normalize_class_name(p.get('class_name', 'unknown'))
    if c in gmk and f.get('geometry'): gmk[c].append(TYPE.get(p.get('markup_type'), '?'))
del gt
for c in CL: assert len(gmk[c]) == T['n_gt'][c], (c, len(gmk[c]), T['n_gt'][c])
gwkb = pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))
boxes = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))

def greedy(c, cand):
    """Официальный жадный проход по кандидатам cand (глоб. индексы, уже в порядке убывания conf)."""
    P, J, S = T['pairs'][(c, c)]
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

cls, conf, reg, grp, mk = T['cls'], T['conf'], T['reg'], T['grp'], T['mk']
order = {}; curves = {}; gmatch_all = {}
for k, c in enumerate(CL):
    idx = np.flatnonzero(cls == k)
    o = idx[np.argsort(-conf[idx], kind='stable')]
    order[c] = o
    m = greedy(c, o); gmatch_all[c] = m
    hit = np.fromiter((i in m for i in o), bool, len(o))
    curves[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                           gt_regions=collections.Counter(T['greg'][c]), pred_group=list(grp[o]))
print('curves', time.time() - t0, flush=True)
regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10)
counts, info = honest_counts(curves, folds)
honest_A = score_counts(counts)[0]
print('honest A (common pass)', honest_A, info, flush=True)
# all-data sweep (in-sample)
res_all = sweep_groups(curves)
# kept mask under honest thresholds (same logic as honest_counts, unseen=drop)
kept = np.zeros(len(cls), bool)
every = set().union(*folds)
for held in folds:
    res = sweep_groups({c: curve_where(cv, every - held) for c, cv in curves.items()})
    for c in CL:
        taus = {g: gv['threshold'] for g, gv in res.get(c, {'groups': {}})['groups'].items()}
        o = order[c]; r = reg[o]; inh = np.isin(r, list(held))
        tau = np.array([taus.get(g, np.inf) for g in grp[o]])
        kept[o[inh & (conf[o] >= tau)]] = True
print('kept', kept.sum(), flush=True)

def table_for(selmask, label):
    """Точный пересчёт (жадный проход только по оставленным) → счётчики по классу×типу."""
    rows = {}; tot = {}; gm = {}
    for c in CL:
        o = order[c][selmask[order[c]]]
        m = greedy(c, o); gm[c] = m
        tpg = np.zeros(T['n_gt'][c], bool); tpg[list(m.values())] = True
        n = T['n_gt'][c]; tp = int(tpg.sum()); fp = len(o) - tp; fn = n - tp
        tot[c] = dict(tp=tp, fp=fp, fn=fn, n_gt=n, kept=len(o), f1=float(_f1(tp, fp, fn)))
        gtype = np.array(gmk[c]); ptype = np.array([TYPE.get(x, '?') for x in mk[o]])
        fpmask = np.array([i not in m for i in o], bool)
        for ty in ['Li', 'Or', 'Ae', 'Geo']:
            ng = int((gtype == ty).sum()); tpt = int((tpg & (gtype == ty)).sum()); fnt = ng - tpt
            fpt = int((fpmask & (ptype == ty)).sum()); tpp = int(((~fpmask) & (ptype == ty)).sum())
            if ng == 0 and fpt == 0 and tpp == 0: continue
            f1 = float(_f1(tp, fp, fn))
            lfn = W[c] / 100 * (float(_f1(tp + fnt, fp, fn - fnt)) - f1)
            lfp = W[c] / 100 * (float(_f1(tp, fp - fpt, fn)) - f1)
            rows[(c, ty)] = dict(n_gt=ng, tp=tpt, fn=fnt, fp=fpt, tp_by_pred=tpp, recall=tpt / ng if ng else None,
                                 f1_type=float(_f1(tpt, fpt, fnt)) if ng else 0.0, loss_fn=lfn, loss_fp=lfp)
        t = tot[c]; f1 = t['f1']
        t['loss_fn'] = W[c] / 100 * (float(_f1(tp + fn, fp, 0)) - f1)
        t['loss_fp'] = W[c] / 100 * (float(_f1(tp, 0, fn)) - f1)
        t['loss_total'] = W[c] / 100 * (1 - f1)
    score = sum(W[c] * tot[c]['f1'] for c in CL) / 100
    return score, rows, tot, gm

sA, rowsA, totA, gmA = table_for(kept, 'A')
print('honest A exact rematch', sA, flush=True)
# region split: lidar vs no lidar (honest kept)
def region_split(gm, selmask):
    out = {}
    for c in CL:
        o = order[c][selmask[order[c]]]; m = gm[c]
        gl = np.array([r in boxes for r in T['greg'][c]]); tpg = np.zeros(T['n_gt'][c], bool); tpg[list(m.values())] = True
        pl = np.isin(reg[o], list(boxes)); fpm = np.array([i not in m for i in o], bool)
        tp, fp, fn = int(tpg.sum()), int(fpm.sum()), int((~tpg).sum()); f1 = float(_f1(tp, fp, fn))
        for name, gsel, psel in [('lidar', gl, pl), ('no_lidar', ~gl, ~pl)]:
            fnt = int(((~tpg) & gsel).sum()); fpt = int((fpm & psel).sum())
            out[(c, name)] = dict(n_gt=int(gsel.sum()), tp=int((tpg & gsel).sum()), fn=fnt, fp=fpt,
                                  loss_fn=W[c] / 100 * (float(_f1(tp + fnt, fp, fn - fnt)) - f1),
                                  loss_fp=W[c] / 100 * (float(_f1(tp, fp - fpt, fn)) - f1))
    return out
splitA = region_split(gmA, kept)
# ceiling 1a: per-class single GT-aware threshold (prefix of the common pass = exact)
c1 = {}
for c in CL:
    cv = curves[c]; tp = np.cumsum(cv.hit); n = np.arange(1, len(tp) + 1)
    last = np.r_[np.flatnonzero(cv.conf[1:] != cv.conf[:-1]), len(tp) - 1] if len(tp) else np.array([], int)
    f = _f1(tp[last], n[last] - tp[last], cv.n_gt - tp[last]) if len(last) else np.array([0.0])
    j = int(np.argmax(f)); c1[c] = dict(f1=float(f[j]), tau=float(cv.conf[last[j]]) if len(last) else None)
# ceiling 1b: per-(class,group) GT-aware thresholds (all-data sweep) — exact rematch
selB = np.zeros(len(cls), bool)
for k, c in enumerate(CL):
    taus = {g: gv['threshold'] for g, gv in res_all.get(c, {'groups': {}})['groups'].items()}
    o = order[c]; tau = np.array([taus.get(g, np.inf) for g in grp[o]]); selB[o[conf[o] >= tau]] = True
sB, rowsB, totB, _ = table_for(selB, 'B')
# ceilings 2/3: oracle reranker (max bipartite matching over same-class pairs) and candidate recall
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
orc = {}
for c in CL:
    P, J, S = T['pairs'][(c, c)]; n = T['n_gt'][c]
    gtype = np.array(gmk[c]); cov = np.zeros(n, bool); cov[J] = True
    if len(P):
        up, pi = np.unique(P, return_inverse=True)
        A = csr_matrix((np.ones(len(P)), (J, pi)), shape=(n, len(up)))
        mm = maximum_bipartite_matching(A, perm_type='column'); matched = mm >= 0
    else: matched = np.zeros(n, bool)
    tp = int(matched.sum())
    orc[c] = dict(n_gt=n, covered=int(cov.sum()), recall_cand=cov.sum() / n, oracle_tp=tp, oracle_f1=float(_f1(tp, 0, n - tp)),
                  by_type={ty: dict(n_gt=int((gtype == ty).sum()), covered=int((cov & (gtype == ty)).sum()), oracle_tp=int((matched & (gtype == ty)).sum()))
                           for ty in ['Li', 'Or', 'Ae', 'Geo'] if (gtype == ty).any()})
s_orc = sum(W[c] * orc[c]['oracle_f1'] for c in CL) / 100
s_c1 = sum(W[c] * c1[c]['f1'] for c in CL) / 100
# near-miss / size analysis on honest A
cx, cy, area = T['cx'], T['cy'], T['area']
near = {}
for k, c in enumerate(CL):
    G = shapely.from_wkb(gwkb[c]); n = len(G)
    if n == 0: continue
    gr = np.array(T['greg'][c]); ga = shapely.area(G); gx, gy = T['gx'][c], T['gy'][c]
    isp = np.array([g.geom_type == 'Point' for g in G])
    tpg = np.zeros(n, bool); tpg[list(gmA[c].values())] = True
    o = order[c]; ok = kept[o]
    keptset = o[ok]; fpk = np.array([i not in gmA[c] for i in keptset], bool)
    # spatial index on candidate centroids per class
    pts = shapely.points(cx[o], cy[o]); tree = shapely.STRtree(pts)
    fn_idx = np.flatnonzero(~tpg)
    covg = np.zeros(n, bool); covg[T['pairs'][(c, c)][1]] = True
    stats = collections.Counter(); area_ratio = []; worth_fp = 0
    used_fp = set()
    for j in fn_idx:
        g = G[j]; geo = g.buffer(5.0) if isp[j] else g.buffer(5.0)
        q = tree.query(geo)
        q = [x for x in q if reg[o[x]] == gr[j]]
        if not q: stats['no_candidate_5m'] += 1; continue
        inside = [x for x in q if (g.contains(pts[x]) if not isp[j] else True)]
        anyk = [x for x in q if ok[x]]
        stats['any_candidate_5m'] += 1
        if inside: stats['cand_centroid_inside_gt'] += 1
        if covg[j]: stats['near_and_matchable(filtering)'] += 1
        else:
            stats['near_but_no_match(geometry)'] += 1
            if inside: stats['inside_but_no_match(geometry)'] += 1
            if isp[j]:
                dmin = min(shapely.distance(pts[x], g) for x in q); stats['pt_nearest_%s' % ('2-3m' if dmin <= 3 else '3-5m' if dmin <= 5 else '>5m')] += 1
        if anyk:
            stats['kept_pred_5m'] += 1
            fpx = [x for x in anyk if o[x] in set() or (o[x] not in gmA[c])]
            if fpx:
                stats['kept_FP_5m'] += 1
                x = fpx[0]
                if o[x] not in used_fp: used_fp.add(o[x]); worth_fp += 1
                if not isp[j]: area_ratio.append(area[o[x]] / max(ga[j], 1e-9))
    tp, fp, fn = totA[c]['tp'], totA[c]['fp'], totA[c]['fn']; f1 = totA[c]['f1']
    gain = W[c] / 100 * (float(_f1(tp + worth_fp, fp - worth_fp, fn - worth_fp)) - f1)
    # size distributions: TP pred vs GT, FN GT, kept FP
    tp_pairs = list(gmA[c].items())
    rat_tp = [area[i] / max(ga[j], 1e-9) for i, j in tp_pairs if not isp[j]]
    q = lambda a: [round(float(x), 2) for x in np.percentile(a, [10, 25, 50, 75, 90])] if len(a) else None
    ng = stats['near_but_no_match(geometry)']; ni = stats['inside_but_no_match(geometry)']
    otp = orc[c]['oracle_tp']
    geo_gain = W[c] / 100 * (float(_f1(otp + ng, 0, n - otp - ng)) - orc[c]['oracle_f1'])
    geo_gain_in = W[c] / 100 * (float(_f1(otp + ni, 0, n - otp - ni)) - orc[c]['oracle_f1'])
    near[c] = dict(oracle_gain_if_geometry_fixed_5m=geo_gain, oracle_gain_if_geometry_fixed_inside=geo_gain_in,n_fn=int(len(fn_idx)), **{k2: int(v) for k2, v in stats.items()}, fp_fixable_unique=worth_fp,
                   gain_if_fixed=gain,
                   gt_area_all_p=q(ga[~isp]), gt_area_TP_p=q(ga[tpg & ~isp]), gt_area_FN_p=q(ga[~tpg & ~isp]),
                   pred_area_TP_p=q(area[[i for i, _ in tp_pairs]]), pred_area_FP_p=q(area[keptset[fpk]]),
                   ratio_TP_pred_to_gt_p=q(rat_tp), ratio_nearFP_to_FNgt_p=q(area_ratio),
                   gt_points=int(isp.sum()))
    print('near', c, near[c], flush=True)
J = lambda d: {f'{a}|{b}': v for (a, b), v in d.items()}
out = dict(honest_A_common_pass=honest_A, honest_A_exact=sA, info=info, totA=totA, rowsA=J(rowsA), splitA=J(splitA),
           ceil1_class=c1, ceil1_class_score=s_c1, ceil1_group_exact=sB, totB=totB, rowsB=J(rowsB),
           oracle=orc, oracle_score=s_orc, near=near, weights=W, n_cand=int(len(cls)), n_kept=int(kept.sum()))
json.dump(out, open(OUT / 'res_gap.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('DONE', time.time() - t0, 'A', honest_A, sA, 'c1', s_c1, 'c1g', sB, 'oracle', s_orc)
