#!/usr/bin/env python3
"""W4 C1F re-gate (PREREG e6b5a31): nested per-fold choice of (d0, s) and sp_dfine τ, non-sp_dfine τ frozen at taus_V3P."""
exec(open("/workspace/koz3/orch/bot-W4/work/q6/q6_prep_head.py").read())
OUT = Path("/workspace/koz3/w4/c1f/out"); OUT.mkdir(parents=True, exist_ok=True)
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
DF = "sp_dfine"


def curve_cls(c, d0, s):
    """one-class curve with the scene-centre prior on sp_dfine extras (same greedy as curves_from)."""
    b, e = base3.base[c], ex[c]
    m = np.ones(len(e["conf"]))
    if s > 0:
        dr = e["drel"]; ok = np.isfinite(dr) & (dr > d0) & (e["grp"] == DF)
        m[ok] = np.maximum(0.0, 1.0 - s * np.minimum((dr[ok] - d0) / (1.0 - d0), 1.0))
    one = curves_from({c: dict(e, conf=e["conf"] * m)}, eng=_Eng1(c))
    return one[c]


class _Eng1:
    def __init__(self, c):
        self.CL = [c]; self.base = base3.base; self.G = base3.G; self.greg = base3.greg


def sweep(curve, taus, regions):
    """argmax of the class F1 over sp_dfine cut points (highest τ on ties); returns τ (1.01 = none)."""
    gnames = curve.pred_group; rg = curve.pred_region
    inr = np.array([r in regions for r in rg], bool)
    isdf = np.array([g == DF for g in gnames], bool)
    tval = np.array([taus.get(g, {}).get(cls_, math.inf) if g != DF else 0 for g in gnames], float)
    keep0 = inr & ~isdf & (curve.conf >= tval)
    tp0 = int(curve.hit[keep0].sum()); fp0 = int(keep0.sum()) - tp0
    G_ = sum(int(k) for r, k in curve.gt_regions.items() if r in regions)
    idx = np.flatnonzero(inr & isdf)          # already sorted by conf desc
    cf = curve.conf[idx]; h = curve.hit[idx].astype(int)
    tp = tp0 + np.concatenate([[0], np.cumsum(h)]); fp = fp0 + np.concatenate([[0], np.cumsum(1 - h)])
    f1 = np.where(2 * tp + fp + (G_ - tp) > 0, 2 * tp / np.maximum(2 * tp + fp + (G_ - tp), 1), 0.0)
    ks = [0] + [k for k in range(1, len(cf) + 1) if k == len(cf) or cf[k - 1] > cf[k]]
    best = max(ks, key=lambda k: (round(float(f1[k]), 12), -k))
    return 1.01 if best == 0 else float(cf[best - 1])


C1GRID = [(0.0, 0.0)] + [(d0, s) for d0 in (0.0, 0.25, 0.5, 0.75) for s in (0.25, 0.5, 0.75)]
P2D0, P2S = (0.0, 0.25, 0.5), [0.25 * k for k in range(7)]
cache = {}
def cv(c, d0, s):
    k = (c, d0 if s > 0 else 0.0, s)
    if k not in cache:
        cache[k] = cB[c] if s == 0 else curve_cls(c, d0, s)
    return cache[k]


def fold_eval(fi, sfun, d0):
    """per class: chosen τ on training, curves; returns taus dict, train score, curves dict."""
    tr = every - folds[fi]; t = {g: dict(d) for g, d in TB[fi].items()}; t.setdefault(DF, {})
    cur = dict(cB)
    for c in CL5:
        global cls_
        cls_ = c
        cu = cv(c, d0, sfun(c)); cur[c] = cu
        t[DF][c] = sweep(cu, t, tr)
    return t, M.score_counts(counts_at(cur, t, tr))[0], cur


res = {"gate_base": sB, "variants": {}}
for name in ("C1", "P2"):
    chosen, taus, held = [], [], {cls: {} for cls in cB}
    for fi in range(len(folds)):
        cands = []
        if name == "C1":
            for d0, s in C1GRID:
                t, sc, cur = fold_eval(fi, lambda c: s, d0)
                cands.append(((round(sc, 12), -s, d0), dict(d0=d0, s_by_class={c: s for c in CL5}), t, cur))
        else:
            for d0 in P2D0:
                for sg in P2S:
                    for sa in P2S:
                        sf = lambda c: sg if c == "gorodishcha" else (sa if c == "arkhitektury" else 0.0)
                        t, sc, cur = fold_eval(fi, sf, d0)
                        cands.append(((round(sc, 12), -(sg + sa), -sg, d0), dict(d0=d0, s_by_class={"gorodishcha": sg, "arkhitektury": sa}), t, cur))
        key, hp, t, cur = max(cands, key=lambda x: x[0])
        for cls, d in counts_at(cur, t, folds[fi]).items():
            held[cls].update(d)
        chosen.append(hp); taus.append(fin(t))
        log(f"{name} fold {fi}: {hp} train {key[0]:.6f} τ_dfine {t[DF]}")
    sg_, per = M.score_counts(held)
    log(f"{name} gate nested (frozen) {sg_:.6f} (base {sB:.6f}) Δ {sg_ - sB:+.6f}")
    res["variants"][name] = dict(chosen=chosen, taus=taus, gate_nested=sg_, cls={c: x["f1"] for c, x in per.items()})
res["taus_base"] = [fin(t) for t in TB]
pickle.dump(dict(res=res, ex_base=ex_c19 + ex_dfx), open(OUT / "prep.pkl", "wb"))
json.dump(res, open(OUT / "prep_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
