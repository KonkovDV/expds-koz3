"""v3a CPU levers on OOF A (PREREG orch/bot-W4/PREREG_v3a_levers.md). Measure-only transforms over work/table.pkl.
Run from wt/w5/analysis/2026-10-30/w5_rerank with W5_DATA/W5_ROOT set (uses its common/fast).
  python v3a_levers.py --levers L10o,L14,L14s,L3m,L3a --outdir /workspace/koz3/orch/bot-W4/runs
  python v3a_levers.py --stack 'L10o=30;L14=0.5' ...   (gate a composed stack once)"""
import sys, os, json, time, argparse
sys.path.insert(0, os.getcwd())
from common import *
import fast, numpy as np, shapely
from expds.metric import ClassCurve
ap = argparse.ArgumentParser(); ap.add_argument('--levers', default=''); ap.add_argument('--stack', default='')
ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--cboot', type=int, default=1000); ap.add_argument('--pl', default='', help='rp: M_rp pairs (W0-seams rp disk) for gor/arch/obj/fort'); ap.add_argument('--tag', default=''); ap.add_argument('--outdir', default='/workspace/koz3/orch/bot-W4/runs')
a = ap.parse_args()
T = fast.T; CL = fast.CL; cid = fast.cid; N = len(T['cls'])
cls0 = T['cls'].astype(int); conf0 = T['conf'].astype(float); grp0 = T['grp'].astype(str); reg0 = T['reg'].astype(str)
MK = np.char.lower(T['mk'].astype(str)); bw, bh, cx, cy, area = T['bw'], T['bh'], T['cx'], T['cy'], T['area']
fam = np.where(cls0 <= 1, 0, cls0)
if a.pl == 'rp':
    NP = pickle.load(open(W/'work/w0_disk_pairs.pkl', 'rb'))['rp']; RPC = {'gorodishcha', 'arkhitektury', 'object_poly', 'fortifikatsii'}
    def _pl(pairs):
        d = {}
        for p, j, s_ in zip(*pairs): d.setdefault(int(p), []).append((int(j), float(s_)))
        return d
    fast.PL = {k: (_pl(NP[k]) if k in NP else {}) if k[0] in RPC else v for k, v in fast.PL.items()}

def curves(grp, keep, src=None, conf=None):
    """grp/keep per emission; src = table row of each emission (default identity: emission i = row i)."""
    src = np.arange(N) if src is None else src
    out = {}
    ecls = cls0[src]; econf = (conf0 if conf is None else conf)[src]
    for c in CL:
        idx = np.flatnonzero((ecls == cid[c]) & keep); o = idx[np.argsort(-econf[idx], kind='stable')]
        hit = np.zeros(len(o), bool); matched = set()
        for r, e in enumerate(o):
            p = src[e]; lst = fast.PL.get((CL[cls0[p]], c), {}).get(int(p))
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r] = True
        out[c] = ClassCurve(conf=econf[o], hit=hit, pred_region=list(reg0[src[o]]), n_gt=T['n_gt'][c], gt_regions=fast.GREG[c], pred_group=list(grp[o]))
    return out

gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt)
# site modality for composition gate (#12): priority geo > li > sp_or > ae > or, from GT layers ∪ pred layers
SL = {}
for f in gt['features']:
    p = f['properties']; SL.setdefault(p['region_name'], set()).add(p['markup_type'])
for r_, m_ in set(zip(reg0.tolist(), MK.tolist())):
    if m_: SL.setdefault(r_, set()).add(m_)
del gt
def site_mod(r):
    s = SL.get(r, set())
    for m, ls in (('geo', {'maggr', 'etgr', 'terfl'}), ('li', {'li'}), ('sp_or', {'sp_or'}), ('ae', {'ae'}), ('or', {'or'})):
        if s & ls: return m
    return 'other'
COMP = {'geo': 0.50, 'li': 0.33, 'sp_or': 0.127, 'ae': 0.026, 'or': 0.017}
CLOSED = {'geo': 0.25, 'li': 0.10, 'sp_or': 0.25, 'ae': 0.25, 'or': 0.15}
PREDL = {}   # pred layers per site (test-time knowable)
for r_, m_ in set(zip(reg0.tolist(), MK.tolist())):
    if m_: PREDL.setdefault(r_, set()).add(m_)

base_keep = cls0 >= 0
cvB = curves(grp0, base_keep)
regions = {r for c in cvB.values() for r in c.pred_region} | {r for c in cvB.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10); K = len(folds)
SM = {r: site_mod(r) for r in regions}
share = {m: sum(1 for r in regions if SM[r] == m) / len(regions) for m in COMP}
WCOMP = {r: (COMP.get(SM[r], 0.0) / share[SM[r]] if share.get(SM[r]) else 0.0) for r in regions}
WCLOS = {r: (CLOSED.get(SM[r], 0.0) / share[SM[r]] if share.get(SM[r]) else 0.0) for r in regions}
FAMS = sorted({site_family(r) for r in regions}); FREG = {f: [r for r in regions if site_family(r) == f] for f in FAMS}
def wboot(cnt, base, Wt, n=2000, seed=0):
    rng = np.random.default_rng(seed); ds = []
    for _ in range(n):
        m = np.bincount(rng.integers(0, len(FAMS), len(FAMS)), minlength=len(FAMS)); w = {}
        for f, k in zip(FAMS, m):
            if k:
                for r in FREG[f]: w[r] = k * Wt[r]
        ds.append(score_counts(cnt, weights=w)[0] - score_counts(base, weights=w)[0])
    ds = np.array(ds); return dict(mean=float(ds.mean()), p5=float(np.percentile(ds, 5)), p95=float(np.percentile(ds, 95)), p_gt0=float((ds > 0).mean()))

def run(grp, keep, nested=True, src=None, conf=None):
    cv = curves(grp, keep, src, conf); full, _ = honest_counts(cv, folds)
    inner = [score_counts(honest_counts(cv, [f for j, f in enumerate(folds) if j != k])[0])[0] for k in range(K)] if nested else None
    return full, inner
def merge(per, pick):
    out = {c: {} for c in CL}
    for k, held in enumerate(folds):
        for c, by in per[pick[k]][0].items():
            for r, v in by.items():
                if r in held: out[c][r] = v
    return out
base_counts = pickle.load(open(W/'work/base_counts.pkl', 'rb'))['counts']
A = score_counts(base_counts)[0]; An = score_counts(base_counts, new)[0]; Ao = score_counts(base_counts, optical)[0]
Ac = score_counts(base_counts, weights=WCOMP)[0]; Acl = score_counts(base_counts, weights=WCLOS)[0]
print('base', A, 'comp', Ac, 'site modality counts', {m: sum(1 for r in regions if SM[r] == m) for m in set(SM.values())}, flush=True)
FLOOR = {'kurgany_povrezhdennye': -0.002, 'kurgany_tselye': -0.002, 'gorodishcha': -0.002}
NGT = {c: sum(v[2] for v in base_counts[c].values()) for c in CL}
def mod_scores(cnt):
    return {m: score_counts(cnt, {r for r in regions if SM[r] == m})[0] for m in ('geo', 'li', 'sp_or', 'ae', 'or')}
BM = mod_scores(base_counts)
def report(res, name, cnt, extra):
    s = score_counts(cnt)[0]; bs = paired_bootstrap_counts(cnt, base_counts, site_family, n=a.boot); bs.pop('samples', None)
    pb = score_counts(base_counts)[1]; pcs = score_counts(cnt)[1]
    pc = {c: round(pcs[c]['f1'] - pb[c]['f1'], 5) for c in pcs if c in pb}
    floors = all(pc[c] >= FLOOR.get(c, -0.005) for c in pc if NGT.get(c, 0) >= 5)
    dm = {m: round(v - BM[m], 5) for m, v in mod_scores(cnt).items()}
    r = dict(score=s, d=s - A, boot=bs, d_new34=score_counts(cnt, new)[0] - An, d_optical=score_counts(cnt, optical)[0] - Ao,
             d_comp=score_counts(cnt, weights=WCOMP)[0] - Ac, boot_comp=wboot(cnt, base_counts, WCOMP, n=a.cboot), d_closed=score_counts(cnt, weights=WCLOS)[0] - Acl, boot_closed=wboot(cnt, base_counts, WCLOS, n=a.cboot), d_by_site_modality=dm, per_class_dF1=pc, floors_ok=floors, **extra)
    r['accept'] = bool(r['d'] >= 0.003 and bs.get('p_gt0', 0) >= 0.9 and r['d_new34'] >= -0.001 and r['d_optical'] >= -0.001 and floors)
    res['variants'][name] = r
    print(name, json.dumps({k: (round(r[k], 5) if isinstance(r[k], float) else r[k]) for k in ('d', 'd_new34', 'd_optical', 'd_comp', 'd_closed', 'accept')}), 'P', bs.get('p_gt0'), 'Ppub', r['boot_comp']['p_gt0'], 'Pcl', r['boot_closed']['p_gt0'], dm, flush=True)

# ---------- levers ----------
def L10o(X):
    keep = base_keep & ~((cls0 == cid['object_poly']) & (np.maximum(bw, bh) > X)); return grp0, keep
_pairs = {}
def dedup_pairs():
    if 'd' in _pairs: return _pairs['d']
    t = time.time(); idx = np.flatnonzero(base_keep)
    boxes = shapely.box(cx[idx] - bw[idx] / 2, cy[idx] - bh[idx] / 2, cx[idx] + bw[idx] / 2, cy[idx] + bh[idx] / 2)
    tree = shapely.STRtree(boxes); I, J = tree.query(boxes, predicate='intersects')
    m = I < J; I, J = idx[I[m]], idx[J[m]]
    m = (reg0[I] == reg0[J]) & (fam[I] == fam[J]); I, J = I[m], J[m]
    ix = np.maximum(0, np.minimum(cx[I] + bw[I] / 2, cx[J] + bw[J] / 2) - np.maximum(cx[I] - bw[I] / 2, cx[J] - bw[J] / 2))
    iy = np.maximum(0, np.minimum(cy[I] + bh[I] / 2, cy[J] + bh[J] / 2) - np.maximum(cy[I] - bh[I] / 2, cy[J] - bh[J] / 2))
    inter = ix * iy; iou = inter / (bw[I] * bh[I] + bw[J] * bh[J] - inter + 1e-9)
    m = iou >= 0.3; I, J, iou = I[m], J[m], iou[m]
    # orient: hi = higher conf (stable tie by index)
    sw = (conf0[J] > conf0[I]) | ((conf0[J] == conf0[I]) & (J < I)); hi = np.where(sw, J, I); lo = np.where(sw, I, J)
    same = grp0[hi] == grp0[lo]
    print('dedup pairs', len(hi), 'same-group', int(same.sum()), round(time.time() - t), 's', flush=True)
    _pairs['d'] = (hi, lo, iou, same); return _pairs['d']
def dedup(t, same_group):
    hi, lo, iou, same = dedup_pairs(); m = (iou >= t) & (same if same_group else ~same); hi, lo = hi[m], lo[m]
    order = np.argsort(-conf0[lo], kind='stable')  # not needed for correctness; greedy over all by conf desc below
    sup = np.zeros(N, bool)
    o = np.argsort(lo, kind='stable'); lo_s, hi_s = lo[o], hi[o]; starts = np.searchsorted(lo_s, np.arange(N)); ends = np.searchsorted(lo_s, np.arange(N), 'right')
    cand = np.unique(lo_s); cand = cand[np.lexsort((cand, -conf0[cand]))]
    for p in cand:   # conf-desc greedy: p suppressed iff some kept higher-conf neighbour
        hs = hi_s[starts[p]:ends[p]]
        if (~sup[hs]).any(): sup[p] = True
    return grp0, base_keep & ~sup
def L14(t): return dedup(t, False)
def L14g(t):
    g, k = dedup(t, False); op = cls0 == cid['object_poly']; return g, np.where(op, k, base_keep)
def L14s(t): return dedup(t, True)
def L3m(v):
    nl = np.array(['li' not in PREDL.get(r, ()) for r in reg0]); g = np.where(nl, np.char.add(grp0, '|nl'), grp0)
    if v == 'nl+aeonly':
        aeo = np.array([(PREDL.get(r, set()) & {'ae', 'sp_or', 'or', 'li'}) == {'ae'} for r in reg0]); g = np.where(aeo, np.char.add(grp0, '|ae1'), g)
    return g, base_keep
BIG = np.isin(cls0, [cid[c] for c in ('gorodishcha', 'fortifikatsii', 'object_poly', 'arkhitektury')])
def L3a(v):
    edges = [5000.0] if v == '5k' else [1000.0, 5000.0, 10000.0]
    b = np.digitize(area, edges).astype(str); g = np.where(BIG, np.char.add(np.char.add(grp0, '|a'), b), grp0); return g, base_keep

XDIR = {'li': ('ae', 'sp_or'), 'sp_or': ('or',), 'or': ('sp_or',)}
def X1(q):
    """directional cross-layer copies of confident preds (conf >= q-quantile of its class×group), own group."""
    key = np.char.add(np.char.add(grp0, '|'), cls0.astype(str)); qthr = {}
    for k in np.unique(key):
        m = key == k; qthr[k] = np.quantile(conf0[m], q)
    thr = np.array([qthr[k] for k in key]); conf_ok = base_keep & (conf0 >= thr)
    s_, t_ = [], []
    for p in np.flatnonzero(conf_ok & np.isin(MK, list(XDIR))):
        for L in XDIR[MK[p]]:
            if L in PREDL.get(reg0[p], ()): s_.append(p); t_.append(L)
    s_ = np.array(s_, int); t_ = np.array(t_, dtype=str)
    src = np.r_[np.arange(N), s_]; g = np.r_[grp0, np.char.add(np.char.add(grp0[s_], '~x>'), t_)]
    keep = np.r_[base_keep, np.ones(len(s_), bool)]; print('X1 copies', len(s_), flush=True)
    return g, keep, src

def L3be_counts():
    """break-even keep: isotonic p(hit|conf) per (class, group) on train sites, keep iff p > F/(4-F) (F = train sweep F1)."""
    from sklearn.isotonic import IsotonicRegression
    from expds.metric import curve_where, sweep_groups
    cv = curves(grp0, base_keep); counts = {c: {} for c in CL}; nfit = 0
    for held in folds:
        train = set(regions) - held
        tr = {c: curve_where(v, train) for c, v in cv.items()}; sw = sweep_groups(tr)
        for c, v in cv.items():
            per = counts[c]
            for r_, k in v.gt_regions.items():
                if r_ in held: per.setdefault(r_, [0, 0, 0])[2] += int(k)
            if c not in sw: continue
            F = sw[c]['f1']; be = F / (4 - F); taus = sw[c]['groups']
            pg = np.array(v.pred_group); pr = np.array(v.pred_region, dtype=object)
            mtr = np.array([r_ in train for r_ in pr]) if len(pr) else np.zeros(0, bool); mho = np.array([r_ in held for r_ in pr]) if len(pr) else np.zeros(0, bool)
            for g in np.unique(pg[mho]) if mho.any() else []:
                gh = mho & (pg == g); gt_ = mtr & (pg == g)
                if g not in taus: continue
                if v.hit[gt_].sum() >= 5:
                    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(v.conf[gt_], v.hit[gt_].astype(float))
                    kept = iso.predict(v.conf[gh]) > be; nfit += 1
                else:
                    kept = v.conf[gh] >= taus[g]['threshold']
                for r_, h in zip(pr[gh][kept], v.hit[gh][kept]):
                    per.setdefault(r_, [0, 0, 0])[0 if h else 1] += 1
    print('L3be iso fits', nfit, flush=True); return counts

def ID(p): return grp0, base_keep
_g1 = {}
def g1_scores():
    if 's' in _g1: return _g1['s']
    E = '/workspace/koz3/out/w5-dino-emb'; cidv = np.load(f'{E}/cid.npy'); ok = np.load(f'{E}/ok.npy'); emb = np.load(f'{E}/emb.npy', mmap_mode='r')
    fold_of = {r: i for i, f in enumerate(folds) for r in f}
    hit = np.zeros(N, bool)
    for c in ('gorodishcha', 'fortifikatsii'):
        P = T['pairs'].get((c, c))
        if P is not None: hit[np.unique(P[0])] = True
    S = np.full(N, np.nan); BR = np.full(N, np.nan); _g1['b'] = BR
    for c in ('gorodishcha', 'fortifikatsii'):
        sel = np.flatnonzero(ok & (cls0[cidv] == cid[c])); rows = cidv[sel]
        X = np.asarray(emb[sel], dtype=np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        fd = np.array([fold_of.get(reg0[r], -1) for r in rows]); y = hit[rows]
        for k in range(K):
            q = fd == k; t = (fd != k) & (fd >= 0)
            if not q.any() or t.sum() < 5: continue
            sim = X[q] @ X[t].T; nn = np.argpartition(-sim, 5, axis=1)[:, :5]
            S[rows[q]] = y[t][nn].mean(1); BR[rows[q]] = max(y[t].mean(), 1e-6)
        print('G1', c, 'emb rows', len(rows), 'pos', int(y.sum()), 'scored', int(np.isfinite(S[rows]).sum()), flush=True)
    _g1['s'] = S; return S
def G1n(lam):
    S = g1_scores(); BR = _g1['b']; f = np.isfinite(S)
    m = np.ones(N); m[f] = np.clip(S[f] / BR[f], 0.5, 2.0) ** lam
    return grp0, base_keep, None, conf0 * m
def G1(lam):
    S = g1_scores(); m = np.where(np.isfinite(S), (0.2 + 1.6 * np.nan_to_num(S, nan=0.5)) ** lam, 1.0)
    return grp0, base_keep, None, conf0 * m
GRID = {'ID': (ID, ['-'], []), 'L10o': (L10o, [30.0], [20.0, 50.0]), 'L14': (L14, [0.3, 0.5, 0.7], []), 'L14s': (L14s, [0.5, 0.7], []), 'L14g': (L14g, [0.3, 0.5, 0.7], []),
        'L3m': (L3m, ['nl', 'nl+aeonly'], []), 'L3a': (L3a, ['5k', '1k5k10k'], []), 'X1': (X1, [0.5, 0.9], []), 'G1': (G1, [1.0], [0.5, 2.0]), 'G1n': (G1n, [0.5, 1.0], [])}

if a.levers:
    for L in a.levers.split(','):
        res = {'lever': L, 'base': A, 'base_comp': Ac, 'base_closed': Acl, 'base_by_site_modality': BM, 'variants': {}}
        if L == 'L3be':
            report(res, 'L3be', L3be_counts(), {})
            json.dump(res, open(f'{a.outdir}/v3a_{L}{a.tag}.json', 'w'), indent=1, ensure_ascii=False, default=float); continue
        fn, params, refs = GRID[L]; per = []
        for p in params + refs:
            t = time.time(); out = fn(p); g, k = out[0], out[1]; src = out[2] if len(out) > 2 else None; cf = out[3] if len(out) > 3 else None
            r_ = run(g, k, nested=(p in params and len(params) > 1), src=src, conf=cf)
            report(res, f'{L}={p}', r_[0], {'param': p, 'sec': round(time.time() - t), 'ref_only': p in refs})
            if p in params: per.append(r_)
        if len(params) > 1:
            pick = [int(np.argmax([per[i][1][k] for i in range(len(params))])) for k in range(K)]
            report(res, f'{L}_nested', merge(per, pick), {'picks': [params[i] for i in pick]})
        json.dump(res, open(f'{a.outdir}/v3a_{L}{a.tag}.json', 'w'), indent=1, ensure_ascii=False, default=float)
print('done')
