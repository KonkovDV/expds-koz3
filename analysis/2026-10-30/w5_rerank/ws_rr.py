"""WS-RR1/RR2 + micro-rp resweep (PREREG addendum 30.10 01:20). Nested honest-10 vs A, paired family bootstrap."""
from common import *
import fast, numpy as np
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from expds.metric import curve_where, ClassCurve
T = fast.T; CL = fast.CL; cid = fast.cid; N = len(T['cls'])
cls = T['cls'].astype(int); conf = T['conf'].astype(float); reg = T['reg'].astype(str); grp = T['grp'].astype(str); mk = T['mk'].astype(str)
NP = pickle.load(open(W/'work/w0_disk_pairs.pkl', 'rb'))
def PLof(pairs):
    out = {}
    for k, (P, J, S) in pairs.items():
        d = {}
        for p, j, s in zip(P, J, S): d.setdefault(int(p), []).append((int(j), float(s)))
        out[k] = d
    return out
RPC = {'gorodishcha', 'arkhitektury', 'object_poly', 'fortifikatsii'}
BASE = fast.PL; RP = PLof(NP['rp']); PLM = {k: (RP.get(k, {}) if k[0] in RPC else v) for k, v in BASE.items()}
def curves(S, PL):
    out, H = {}, np.zeros(N, bool)
    for c in CL:
        idx = np.flatnonzero(cls == cid[c]); o = idx[np.argsort(-S[idx], kind='stable')]
        hit = np.zeros(len(o), bool); matched = set()
        for r, p in enumerate(o):
            lst = PL.get((CL[cls[p]], c), {}).get(int(p))
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r] = True
        H[o[hit]] = True
        out[c] = ClassCurve(conf=S[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c], gt_regions=fast.GREG[c], pred_group=list(grp[o]))
    return out, H
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
cv0, _ = curves(conf, BASE)
regions = {r for c in cv0.values() for r in c.pred_region} | {r for c in cv0.values() for r in c.gt_regions}
folds = family_folds(regions, site_family, 10); K = len(folds)
fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in reg])
z = np.load(W/'work/nested_w5cnx.npz'); OUT = z['outer']; INN = z['inner']
li = mk == 'li'
RB = np.zeros(N); LIM = {}
for c in CL:
    m = np.flatnonzero(li & (cls == cid[c]))
    if len(m): RB[m] = (rankdata(conf[m]) - 0.5) / len(m); LIM[c] = m
def blend(k, a):
    R = np.where(fid == k, OUT, INN[k]); S = conf.copy()
    for c, m in LIM.items():
        S[m] = RB[m]; pm = m[R[m] >= 0]
        if len(pm) < 2: continue
        q = np.sort(RB[pm]); rt = q[(rankdata(R[pm], method='ordinal') - 1).astype(int)]
        S[pm] = a * rt + (1 - a) * RB[pm]
    return S
CI = [cid[c] for c in CL]
def evaluate(cv, H, S, held_list, excl, mode, lams=(None,)):
    """returns {lam: counts} over held folds; train = regions - held - excl"""
    outs = {l: {c: {} for c in CL} for l in lams}
    for held in held_list:
        train = regions - held - excl; trm = np.isin(reg, list(train)); hm = np.isin(reg, list(held))
        res = sweep_groups({c: curve_where(v, train) for c, v in cv.items()})
        cal = np.zeros(N, bool); p = np.zeros(N)
        tauv = np.full(N, np.inf)
        for c, v in res.items():
            for g, gv in v['groups'].items():
                mm = hm & (cls == cid[c]) & (grp == g); tauv[mm] = gv['threshold']
        keep_tau = hm & (S >= tauv)
        if mode != 'sweep':
            be = np.zeros(N, bool)
            for c in CL:
                if c not in res: continue
                F = res[c]['f1']; thr = F / (4 - F)
                for L in np.unique(mk[cls == cid[c]]):
                    tr = trm & (cls == cid[c]) & (mk == L)
                    if H[tr].sum() < 5: continue
                    ho = np.flatnonzero(hm & (cls == cid[c]) & (mk == L))
                    if not len(ho): continue
                    ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(S[tr], H[tr].astype(float))
                    p[ho] = ir.predict(S[ho]); cal[ho] = True
                    be[ho] = (p[ho] > thr) & np.isfinite(tauv[ho])
        for l in lams:
            if mode == 'sweep': keep = keep_tau
            elif mode == 'be': keep = np.where(cal, be, keep_tau)
            else:
                keep = np.where(cal, False, keep_tau); ii = np.flatnonzero(cal & np.isfinite(tauv))
                key = np.char.add(np.char.add(reg[ii], '|'), cls[ii].astype(str)); o = np.lexsort((-S[ii], key)); ks = key[o]
                first = np.r_[True, ks[1:] != ks[:-1]]; gid = np.cumsum(first) - 1
                start = np.flatnonzero(first); rank = np.arange(len(o)) - start[gid]
                Kq = np.round(l * np.bincount(gid, weights=p[ii][o])).astype(int)
                keep[ii[o[rank < Kq[gid]]]] = True
            cnt = outs[l]
            for c in CL:
                per = cnt[c]
                for r, n in cv[c].gt_regions.items():
                    if r in held: per.setdefault(r, [0, 0, 0])[2] += int(n)
                mm = np.flatnonzero(keep & (cls == cid[c]))
                for r, h in zip(reg[mm], H[mm]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return outs
def sc(cnt): return score_counts(cnt)[0]
def merge(dst, src, held):
    for c in CL:
        for r, v in src[c].items():
            if r in held: dst[c][r] = v
# sanity: A
cvA, HA = curves(conf, BASE); cA = evaluate(cvA, HA, conf, folds, set(), 'sweep')[None]
A = sc(cA); print('A', A, flush=True); assert abs(A - 0.3146153226828423) < 1e-9
cvM, HM = curves(conf, PLM); cM = evaluate(cvM, HM, conf, folds, set(), 'sweep')[None]
ALPHA = [0.3, 0.6, 1.0]; LAM = [1.0, 1.5, 2.0]
R1 = {c: {} for c in CL}; R1s = {c: {} for c in CL}; R2 = {c: {} for c in CL}; picks = []
for k in range(K):
    t = time.time(); held_in = [f for j, f in enumerate(folds) if j != k]; inn = {}
    for a in ALPHA:
        S = blend(k, a); cv, H = curves(S, PLM)
        inn[('sw', a)] = sc(evaluate(cv, H, S, held_in, folds[k], 'sweep')[None])
        inn[('be', a)] = sc(evaluate(cv, H, S, held_in, folds[k], 'be')[None])
        for l, cnt in evaluate(cv, H, S, held_in, folds[k], 'topk', LAM).items(): inn[('tk', a, l)] = sc(cnt)
    ps = max(ALPHA, key=lambda a: inn[('sw', a)]); pb = max(ALPHA, key=lambda a: inn[('be', a)])
    pt = max([(a, l) for a in ALPHA for l in LAM], key=lambda x: inn[('tk',) + x])
    for dst, a, mode, l in ((R1s, ps, 'sweep', None), (R1, pb, 'be', None), (R2, pt[0], 'topk', pt[1])):
        S = blend(k, a); cv, H = curves(S, PLM)
        merge(dst, evaluate(cv, H, S, [folds[k]], set(), mode, (l,))[l], folds[k])
    picks.append(dict(sw=ps, be=pb, tk=pt)); print('fold', k, picks[-1], f'{time.time()-t:.0f}s', flush=True)
FLOOR = {'kurgany_povrezhdennye': -0.002, 'kurgany_tselye': -0.002, 'gorodishcha': -0.002}
pA = score_counts(cA)[1]; res = {'A': A, 'picks': picks}
for name, cnt in (('M_rp', cM), ('RR1s_ref', R1s), ('RR1', R1), ('RR2', R2)):
    s = sc(cnt); bt = paired_bootstrap_counts(cnt, cA, site_family, n=2000); pc = score_counts(cnt)[1]
    dcl = {c: pc[c]['f1'] - pA[c]['f1'] for c in CL if pA[c]['n_gt'] >= 5}
    fl = all(v >= FLOOR.get(c, -0.005) for c, v in dcl.items())
    r = dict(score=s, d=s - A, P=bt['p_gt0'], p5=bt['p5'], p95=bt['p95'], d_new34=sc_n if False else score_counts(cnt, new)[0] - score_counts(cA, new)[0],
             d_optical=score_counts(cnt, optical)[0] - score_counts(cA, optical)[0], dF1=dcl, floors_ok=fl)
    r['PASS'] = bool(r['d'] >= 0.003 and r['P'] >= 0.9 and fl); res[name] = r
    print(name, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != 'dF1'}), {c[:6]: round(v, 4) for c, v in dcl.items()}, flush=True)
json.dump(res, open('ws_rr_results.json', 'w'), indent=1, default=float)
print('WSRRDONE')
