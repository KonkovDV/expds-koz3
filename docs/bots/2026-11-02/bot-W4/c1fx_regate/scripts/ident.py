#!/usr/bin/env python3
"""W4 GA / FP1 / GAFP (PREREG d366a45): nested gate-side selection under the freeze rule. argv: variants (GA FP1 [GAFP])."""
import sys
VARS = sys.argv[1:] or ["GA", "FP1"]; sys.argv = sys.argv[:1]
exec(open("/workspace/koz3/orch/bot-W4/work/q6/q6_prep_head.py").read())
OUT = Path("/workspace/koz3/w4/c1fx/out"); OUT.mkdir(parents=True, exist_ok=True)
fin = lambda t: {g: {c: (None if not math.isfinite(v) else float(v)) for c, v in d.items()} for g, d in t.items()}
unfin = lambda t: {g: {c: (math.inf if v is None else float(v)) for c, v in d.items()} for g, d in t.items()}
TB = [unfin(t) for t in T["taus_V3P"]]
ex = ex_of(ex_c19 + ex_dfx)
cB = curves_from(ex)
hB = {cls: {} for cls in cB}
for fi in range(len(folds)):
    for cls, d in counts_at(cB, TB[fi], folds[fi]).items():
        hB[cls].update(d)
sB = M.score_counts(hB)[0]; log(f"sanity gate base {sB:.7f} vs {T['V3P']:.7f}"); assert abs(sB - T["V3P"]) < 1e-6
DF = "sp_dfine"; GOR, ARKH = "gorodishcha", "arkhitektury"
GAC = (GOR, ARKH); FPC = ("kurgany_povrezhdennye", "kurgany_tselye", ARKH, "fortifikatsii")


class _Eng1:
    def __init__(self, c):
        self.CL = [c]; self.base = base3.base; self.G = base3.G; self.greg = base3.greg


def mult(dr, grp, d0, s):
    m = np.ones(len(dr))
    if s > 0:
        ok = np.isfinite(dr) & (dr > d0) & (grp == DF)
        m[ok] = np.maximum(0.0, 1.0 - s * np.minimum((dr[ok] - d0) / (1.0 - d0), 1.0))
    return m


cache = {}
def cv(c, pri):
    """pri = tuple of (d0, s) priors applied to class c (multiplied)."""
    pri = tuple(p for p in pri if p[1] > 0)
    if not pri:
        return cB[c]
    if (c, pri) not in cache:
        e = ex[c]; m = np.ones(len(e["conf"]))
        for d0, s in pri:
            m = m * mult(e["drel"], e["grp"], d0, s)
        cache[(c, pri)] = curves_from({c: dict(e, conf=e["conf"] * m)}, eng=_Eng1(c))[c]
    return cache[(c, pri)]


def sweep(curve, cls, taus, regions):
    gnames = curve.pred_group
    inr = np.array([r in regions for r in curve.pred_region], bool)
    isdf = np.array([g == DF for g in gnames], bool)
    tval = np.array([taus.get(g, {}).get(cls, math.inf) if g != DF else 0 for g in gnames], float)
    keep0 = inr & ~isdf & (curve.conf >= tval)
    tp0 = int(curve.hit[keep0].sum()); fp0 = int(keep0.sum()) - tp0
    G_ = sum(int(k) for r, k in curve.gt_regions.items() if r in regions)
    idx = np.flatnonzero(inr & isdf)
    cf = curve.conf[idx]; h = curve.hit[idx].astype(int)
    tp = tp0 + np.concatenate([[0], np.cumsum(h)]); fp = fp0 + np.concatenate([[0], np.cumsum(1 - h)])
    f1 = np.where(tp + fp > 0, M._f1(tp, fp, G_ - tp), 0.0)   # metric F1 (FP weight 0.5)
    ks = [0] + [k for k in range(1, len(cf) + 1) if k == len(cf) or cf[k - 1] > cf[k]]
    best = max(ks, key=lambda k: (round(float(f1[k]), 12), -k))
    return 1.01 if best == 0 else float(cf[best - 1])


def evaluate(tb, tr, pri_of):
    """pri_of(c) → priors tuple; touched classes = those with a prior; returns taus, train score, curves."""
    t = {g: dict(d) for g, d in tb.items()}; cur = dict(cB)
    for c in CL5:
        p = tuple(x for x in pri_of(c) if x[1] > 0)
        if p:
            cur[c] = cv(c, p); t[DF][c] = sweep(cur[c], c, t, tr)
    return t, M.score_counts(counts_at(cur, t, tr))[0], cur


def sweep_old(curve, cls, taus, regions):   # C1F re-gate rule: plain F1, τ = score
    gnames = curve.pred_group
    inr = np.array([r in regions for r in curve.pred_region], bool); isdf = np.array([g == DF for g in gnames], bool)
    tval = np.array([taus.get(g, {}).get(cls, math.inf) if g != DF else 0 for g in gnames], float)
    keep0 = inr & ~isdf & (curve.conf >= tval); tp0 = int(curve.hit[keep0].sum()); fp0 = int(keep0.sum()) - tp0
    G_ = sum(int(k) for r, k in curve.gt_regions.items() if r in regions)
    idx = np.flatnonzero(inr & isdf); cf = curve.conf[idx]; h = curve.hit[idx].astype(int)
    tp = tp0 + np.concatenate([[0], np.cumsum(h)]); fp = fp0 + np.concatenate([[0], np.cumsum(1 - h)])
    f1 = 2 * tp / np.maximum(2 * tp + fp + (G_ - tp), 1)
    ks = [0] + [k for k in range(1, len(cf) + 1) if k == len(cf) or cf[k - 1] > cf[k]]
    best = max(ks, key=lambda k: (round(float(f1[k]), 12), -k)); return 1.01 if best == 0 else float(cf[best - 1])

def sweep_prod(curve, cls, taus, regions):  # metric/prod rule: F1 with FP 0.5, M._group_cuts midpoints, first argmax, round 6
    gnames = curve.pred_group
    inr = np.array([r in regions for r in curve.pred_region], bool); isdf = np.array([g == DF for g in gnames], bool)
    tval = np.array([taus.get(g, {}).get(cls, math.inf) if g != DF else 0 for g in gnames], float)
    keep0 = inr & ~isdf & (curve.conf >= tval); tp0 = float(curve.hit[keep0].sum()); n0 = int(keep0.sum())
    G_ = sum(int(k) for r, k in curve.gt_regions.items() if r in regions)
    idx = np.flatnonzero(inr & isdf); o = np.argsort(-curve.conf[idx], kind="stable")
    keep, tp, cut = M._group_cuts(curve.conf[idx][o], curve.hit[idx][o])
    tt, nn = tp0 + tp, n0 + keep
    f = np.where(nn > 0, M._f1(tt, nn - tt, G_ - tt), 0.0)
    return round(float(cut[int(np.argmax(f))]), 6)

res = {}
for nm, fn in (("old_c1f", sweep_old), ("c1fx_fixedF1", sweep), ("prod", sweep_prod)):
    ch = []
    for fi in range(len(folds)):
        tr = every - folds[fi]
        for c in CL5:
            b = TB[fi][DF][c]; t = fn(cB[c], c, TB[fi], tr)
            if not (abs(t - b) < 5e-7 or (t >= 1.0 and not math.isfinite(b))): ch.append((fi, c, b, t))
    res[nm] = ch; log(nm, "changed vs base", len(ch), "of", len(folds) * len(CL5), ch[:4])
json.dump(res, open("ident_check.json", "w"), default=str, indent=0)
