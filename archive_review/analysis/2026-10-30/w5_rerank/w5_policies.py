"""W5-rerank: post-policies over the stack table, honest-10 gate with nested parameter choice.
Variants (PREREG_w5_rerank.md): V0 base, V1 min-k, V2 kurgan hedge, V3 cross-layer copy, V4 micro-circles, V5 rerank score (from nested2 npz).
All variants are candidate/score transforms; thresholds are swept per (class, group) on train folds only (expds.metric.honest_counts).
Param choice is nested: for outer fold k, the param is picked by 9-fold honest score on the other folds."""
from common import *
import fast, numpy as np, argparse, shapely
from collections import Counter
from scipy.spatial import cKDTree
from expds.metric import ClassCurve
from expds import metric as M
ap = argparse.ArgumentParser(); ap.add_argument('--variants', default='V0,V1,V2,V3,V4'); ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--rerank', default='', help='npz with outer/inner reranker scores (nested2.py) for V5'); ap.add_argument('--out', default='w5_results.json')
ap.add_argument('--nested', type=int, default=1)
a = ap.parse_args()
T = fast.T; CL = fast.CL; cid = fast.cid; N = len(T['cls']); cls0 = T['cls'].astype(int); conf0 = T['conf'].astype(float); grp0 = T['grp'].astype(str); reg0 = T['reg'].astype(str)
KUR = (cid['kurgany_povrezhdennye'], cid['kurgany_tselye'])
SYN = {}   # synthetic id -> {gt class: [(j, s)]}
SYN_REG = []

def curves(src, sc, cl, gp, syn=None):
    """Generalised fast.curves: emission arrays (src row in T or -1, score, class id, group str, synthetic id or -1)."""
    syn = np.full(len(src), -1) if syn is None else syn
    out = {}
    for c in CL:
        idx = np.flatnonzero(cl == cid[c]); o = idx[np.argsort(-sc[idx], kind='stable')]
        hit = np.zeros(len(o), bool); matched = set()
        for r, e in enumerate(o):
            if syn[e] >= 0: lst = SYN.get(int(syn[e]), {}).get(c)
            else:
                p = src[e]; lst = fast.PL.get((CL[cls0[p]], c), {}).get(int(p))
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r] = True
        regs = [reg0[src[e]] if src[e] >= 0 else SYN_REG[syn[e]] for e in o]
        out[c] = ClassCurve(conf=sc[o].astype(np.float64), hit=hit, pred_region=regs, n_gt=T['n_gt'][c], gt_regions=fast.GREG[c], pred_group=list(gp[o]))
    return out

base_m = cls0 >= 0
B = dict(src=np.flatnonzero(base_m), sc=conf0[base_m], cl=cls0[base_m], gp=grp0[base_m])
def cat(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

def v_mink(k):
    sc = B['sc'].copy(); key = np.char.add(np.char.add(reg0[B['src']], '|'), B['cl'].astype(str))
    o = np.lexsort((-sc, key)); ks = key[o]; first = np.r_[True, ks[1:] != ks[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0)); rank = np.arange(len(o)) - start
    top = o[rank < k]; sc[top] += 1.0
    return dict(B, sc=sc)
def v_hedge(alpha):
    m = np.isin(B['cl'], KUR)
    d = dict(src=B['src'][m], sc=B['sc'][m] * alpha, cl=np.where(B['cl'][m] == KUR[0], KUR[1], KUR[0]), gp=np.char.add(B['gp'][m], '~hedge'))
    return cat(B, d)
MK = np.char.lower(T['mk'].astype(str))
# layers present at a site = markup_types the stack emits there (known at test time from the inputs the models ran on)
SITE_LAYERS = {}
for r_, m_ in set(zip(reg0.tolist(), MK.tolist())):
    if m_: SITE_LAYERS.setdefault(r_, set()).add(m_)
def copy_rows():
    """(src, target markup_type) for every candidate x every OTHER layer present at its site."""
    s, t = [], []
    for p in B['src']:
        for L in sorted(SITE_LAYERS.get(reg0[p], ()) - {MK[p]}): s.append(p); t.append(L)
    return np.array(s, int), np.array(t, dtype=str)
_cp = {}
def v_copy(beta):
    if not _cp: _cp['d'] = copy_rows()
    s, t = _cp['d']
    d = dict(src=s, sc=conf0[s] * beta, cl=cls0[s], gp=np.char.add(np.char.add(grp0[s], '~copy>'), t))
    return cat(B, d)

# --- synthetic micro-circles (V4) ---
_mc = {}
def build_micro(r=0.75, gq_min=0.9):
    if _mc: return _mc['d']
    mod = M.load_official(METRIC); calc = mod.F1Calculator()
    gtw = pickle.load(open(W/'work/gt_wkb.pkl', 'rb')); G = {c: shapely.from_wkb(v) for c, v in gtw.items()}
    greg = T['greg']; X = np.load(W/'work/X.npy'); names = json.load(open(W/'work/X_names.json')); gq = X[:, names.index('gq')]
    cx, cy = T['cx'], T['cy']; fam = np.where(cls0 <= 1, 0, cls0)
    rows = []  # (src, x, y, kind)
    for rg in np.unique(reg0):
        ir = np.flatnonzero((reg0 == rg) & (cls0 >= 0))
        for f in np.unique(fam[ir]):
            ii = ir[fam[ir] == f]; strong = ii[gq[ii] >= gq_min]
            if not len(strong): continue
            for p in strong: rows.append((p, cx[p], cy[p], 0))           # shrink: circle at own centre
            tr = cKDTree(np.c_[cx[ii], cy[ii]])
            for p, nb in zip(strong, tr.query_ball_point(np.c_[cx[strong], cy[strong]], 8.0)):
                nb = [ii[j] for j in nb if grp0[ii[j]] != grp0[p]]
                if not nb: continue
                q = max(nb, key=lambda j: conf0[j]); dd = np.hypot(cx[q] - cx[p], cy[q] - cy[p])
                if dd >= 2.0: rows.append((p, (cx[p] + cx[q]) / 2, (cy[p] + cy[q]) / 2, 1))  # centre disagreement: midpoint
    rows = np.array(rows, dtype=object); src = rows[:, 0].astype(int)
    circ = shapely.buffer(shapely.points(rows[:, 1].astype(float), rows[:, 2].astype(float)), r, quad_segs=8)
    KL = ['kurgany_povrezhdennye', 'kurgany_tselye']
    for c in CL:
        gg = G[c]
        if not len(gg): continue
        tree = shapely.STRtree([g.buffer(calc.r_tolerance) if g.geom_type == 'Point' else g for g in gg])
        pcs = [cid[x] for x in (KL if c in KL else [c])]
        sel = np.flatnonzero(np.isin(cls0[src], pcs)); qp, qg = tree.query(circ[sel])
        for i, j in zip(qp, qg):
            e = sel[i]
            ok, s = calc._match_quality(circ[e], gg[j])
            if ok: SYN.setdefault(int(e), {}).setdefault(c, []).append((int(j), float(s)))
    SYN_REG.extend(reg0[src].tolist())
    _mc['d'] = (src, rows[:, 3].astype(int)); return _mc['d']
def v_micro(beta, kinds=(0, 1)):
    src, kind = build_micro(); m = np.isin(kind, kinds); e = np.flatnonzero(m)
    d = dict(src=src[m], sc=conf0[src[m]] * beta, cl=cls0[src[m]], gp=np.char.add(grp0[src[m]], np.where(kind[m] == 0, '~mc0', '~mc1')))
    syn = np.r_[np.full(len(B['src']), -1), e]
    return cat(B, d), syn

gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
cvB = curves(**B)
regions = {r for c in cvB.values() for r in c.pred_region} | {r for c in cvB.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10); K = len(folds)
def run(em, syn=None):
    cv = curves(em['src'], em['sc'], em['cl'], em['gp'], syn)
    full, _ = honest_counts(cv, folds)
    inner = [score_counts(honest_counts(cv, [f for j, f in enumerate(folds) if j != k])[0])[0] for k in range(K)] if a.nested else None
    return full, inner
def merge(per_param, pick):
    """counts for held fold k from param pick[k]"""
    out = {c: {} for c in CL}
    for k, held in enumerate(folds):
        cnt = per_param[pick[k]][0]
        for c, by in cnt.items():
            for r, v in by.items():
                if r in held: out[c][r] = v
    return out
base_counts, base_inner = run(B)
A = score_counts(base_counts)[0]; print('V0 base honest', A, flush=True)
GRID = {'V1': ('mink', [1, 2, 3], v_mink), 'V2': ('hedge', [0.25, 0.5, 0.75], v_hedge), 'V3': ('copy', [0.5, 0.8, 1.0], v_copy), 'V4': ('micro', [0.3, 0.6, 0.9], v_micro)}
res = {'base': A, 'base_new34': score_counts(base_counts, new)[0], 'base_optical': score_counts(base_counts, optical)[0], 'variants': {}}
def report(name, cnt, extra):
    s = score_counts(cnt)[0]; bs = paired_bootstrap_counts(cnt, base_counts, site_family, n=a.boot)
    pb = score_counts(base_counts)[1]; pc = {c: round(v['f1'] - pb[c]['f1'], 4) for c, v in score_counts(cnt)[1].items() if c in pb}
    r = dict(score=s, d=s - A, boot=bs, d_new34=score_counts(cnt, new)[0] - res['base_new34'], d_optical=score_counts(cnt, optical)[0] - res['base_optical'], per_class_dF1=pc, **extra)
    r['accept'] = bool(r['d'] >= 0.003 and (r['boot'] or {}).get('p_gt0', 0) >= 0.9 and r['d_new34'] >= -0.001 and r['d_optical'] >= -0.001)
    res['variants'][name] = r; print(name, json.dumps({k: r[k] for k in ('score', 'd', 'd_new34', 'd_optical', 'accept')}), (r['boot'] or {}).get('p_gt0'), flush=True)
    json.dump(res, open(W/'work'/a.out, 'w'), indent=1, ensure_ascii=False, default=float)
for v in a.variants.split(','):
    if v == 'V0': continue
    if v == 'V5':
        z = np.load(a.rerank); outer = z['outer']; sc = np.where(outer[B['src']] >= 0, outer[B['src']], -1.0)
        cnt, _ = run(dict(B, sc=sc)); report('V5_rerank', cnt, {}); continue
    nm, params, fn = GRID[v]; per = []
    for p in params:
        t = time.time(); r = fn(p); em, syn = r if isinstance(r, tuple) else (r, None)
        per.append(run(em, syn)); report(f'{v}_{nm}={p}', per[-1][0], {'param': p, 'sec': round(time.time() - t)})
    pick = [int(np.argmax([per[i][1][k] for i in range(len(params))])) for k in range(K)]
    report(f'{v}_{nm}_nested', merge(per, pick), {'picks': [params[i] for i in pick]})
print('done', W/'work'/a.out)
