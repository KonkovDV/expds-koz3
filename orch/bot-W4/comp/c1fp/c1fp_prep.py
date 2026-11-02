#!/usr/bin/env python3
"""W4 C1 / GA / FP1 (PREREG d366a45 + AMEND1): nested gate-side selection under the freeze rule. argv: variants (GA FP1 [GAFP])."""
import sys
VARS = sys.argv[1:] or ["GA", "FP1"]; sys.argv = sys.argv[:1]
exec(open("/workspace/koz3/v4build/gate/q6_prep_head.py").read())
OUT = Path("/workspace/koz3/orch/bot-W4/comp/c1fp/out"); OUT.mkdir(parents=True, exist_ok=True)
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
    """AMEND1: metric/prod rule — F1 with FP weight 0.5 (M._f1), cut = M._group_cuts midpoint, first argmax, round 6."""
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


# AMEND1 sanity gate: no prior → 0 sp_dfine τ changes vs base (all 50 reproduced)
_chg = [(fi, c) for fi in range(len(folds)) for c in CL5
        if not abs(sweep(cB[c], c, TB[fi], every - folds[fi]) - TB[fi][DF][c]) < 5e-7]
log("identity sanity: changed", len(_chg), "of", len(folds) * len(CL5)); assert not _chg, _chg


def evaluate(tb, tr, pri_of):
    """pri_of(c) → priors tuple; touched classes = those with a prior; returns taus, train score, curves."""
    t = {g: dict(d) for g, d in tb.items()}; cur = dict(cB)
    for c in CL5:
        p = tuple(x for x in pri_of(c) if x[1] > 0)
        if p:
            cur[c] = cv(c, p); t[DF][c] = sweep(cur[c], c, t, tr)
    return t, M.score_counts(counts_at(cur, t, tr))[0], cur


GAGRID = [(0.0, 0.0)] + [(d0, s) for d0 in (0.0, 0.25, 0.5, 0.75) for s in (0.25, 0.5, 0.75)]
FP = (0.5, 0.5)


def ga_pri(d0, s):
    return lambda c: ((d0, s),) if c in GAC else ()


def c1_pri(d0, s):
    return lambda c: ((d0, s),)


def gafp_pri(d0, s):
    return lambda c: (((d0, s),) if c in GAC else ()) + ((FP,) if c in FPC else ())


def c1fp_pri(d0, s):
    return lambda c: ((d0, s),) + ((FP,) if c in FPC else ())


def choose_ga(tb, tr, pri=ga_pri):
    best = None
    for d0, s in GAGRID:
        t, sc, cur = evaluate(tb, tr, pri(d0, s))
        key = (round(sc, 12), -s, d0)
        if best is None or key > best[0]:
            best = (key, (d0, s), t, cur)
    return best


res = {"gate_base": sB, "variants": {}}
for name in VARS:
    chosen, taus, held = [], [], {cls: {} for cls in cB}
    for fi in range(len(folds)):
        tr = every - folds[fi]
        if name in ("GA", "C1"):
            key, hp, t, cur = choose_ga(TB[fi], tr, ga_pri if name == "GA" else c1_pri); hpd = dict(d0=hp[0], s=hp[1], train=key[0])
        elif name == "FP1":
            t, sc, cur = evaluate(TB[fi], tr, lambda c: (FP,) if c in FPC else ()); hpd = dict(d0=0.5, edge=0.5, train=sc)
        elif name == "C1FP":   # W4 v5: C1 centre (all classes, C1's per-fold nested (d0,s)) × FP1 frame (0.5,0.5) on FPC
            cch = json.load(open(OUT / "prep_summary.json"))["variants"]["C1"]["chosen"][fi]
            t, sc, cur = evaluate(TB[fi], tr, c1fp_pri(cch["d0"], cch["s"])); hpd = dict(center=dict(d0=cch["d0"], s=cch["s"]), frame=dict(d0=0.5, edge=0.5), train=sc)
        else:
            gch = json.load(open(OUT / "prep_summary.json"))["variants"]["GA"]["chosen"][fi]
            t, sc, cur = evaluate(TB[fi], tr, gafp_pri(gch["d0"], gch["s"])); hpd = dict(center=dict(d0=gch["d0"], s=gch["s"]), frame=dict(d0=0.5, edge=0.5), train=sc)
        assert all(t[g] == TB[fi][g] for g in TB[fi] if g != DF), "freeze violated"
        for cls, d in counts_at(cur, t, folds[fi]).items():
            held[cls].update(d)
        chosen.append(hpd); taus.append(fin(t))
        log(f"{name} fold {fi}: {hpd} τ_dfine {({c: round(v, 6) for c, v in t[DF].items()})}")
    sg_, per = M.score_counts(held)
    log(f"{name} gate nested (frozen) {sg_:.6f} (base {sB:.6f}) Δ {sg_ - sB:+.6f}")
    res["variants"][name] = dict(chosen=chosen, taus=taus, gate_nested=sg_, cls={c: x["f1"] for c, x in per.items()})
res["taus_base"] = [fin(t) for t in TB]
old = json.load(open(OUT / "prep_summary.json")) if (OUT / "prep_summary.json").exists() else {"variants": {}}
old["variants"].update(res["variants"]); res["variants"] = old["variants"]
pickle.dump(dict(res=res, ex_base=ex_c19 + ex_dfx), open(OUT / "prep.pkl", "wb"))
json.dump(res, open(OUT / "prep_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
