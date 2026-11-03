#!/usr/bin/env python3
"""Q6 step 3: gate-side nested τ for maggr_fgbm (frozen: other groups at base taus; unfrozen: full retune). PREREG 870433c."""
exec(open("/workspace/koz3/orch/bot-W4/work/q6/q6_prep_head.py").read())
import copy as _copy
ex_fg = [as_ex(f) for f in json.loads(open("/workspace/koz3/orch/bot-W4/work/q6/ex_fgbm.geojson", "rb").read())["features"]]
log("fgbm extras", len(ex_fg))
fin = lambda t: {g: {c: (None if not math.isfinite(v) else float(v)) for c, v in d.items()} for g, d in t.items()}
unfin = lambda t: {g: {c: (math.inf if v is None else float(v)) for c, v in d.items()} for g, d in t.items()}
TB = [unfin(t) for t in T["taus_V3P"]]
ex_b = ex_of(ex_c19 + ex_dfx)
cB = curves_from(ex_b)
nB, ftB, trB = run(cB)
sB = M.score_counts(nB)[0]
log(f"sanity gate base {sB:.7f} vs V3P {T['V3P']:.7f}"); assert abs(sB - T["V3P"]) < 1e-6
assert [fin(t) for t in ftB] == T["taus_V3P"], "base fold taus differ"
ex_v = ex_of(ex_c19 + ex_dfx + ex_fg)
cV = curves_from(ex_v)
FP = "finds_points"
# AMENDMENT A1 (before any variant score): curves_from fixes TP flags by one greedy pass over ALL predictions, so
# fgbm candidates that are later thresholded out still "steal" GT from classic finds (UNI tau=inf gave 0.3294 != 0.3338).
# For finds we therefore recompute the greedy matching exactly after thresholding (same rule as curves_from/compute_metrics).
_e = ex_v[FP]; _b = base3.base[FP]
F_conf = np.concatenate([_b["conf"], _e["conf"]]); F_reg = np.concatenate([_b["reg"], _e["reg"]]).astype(object)
F_grp = np.concatenate([_b["grp"], _e["grp"]]).astype(object); _nb = len(_b["conf"])
F_pairs = dict(_b["pairs"]); F_pairs.update({_nb + i: v for i, v in _e["pairs"].items()})
F_pairs = {i: sorted(v) for i, v in F_pairs.items()}
_ord = np.argsort(-F_conf, kind="stable"); F_order = _ord.tolist()
F_gt = cV[FP].gt_regions


def finds_exact(taus, regions):
    tau = np.array([taus.get(g, {}).get(FP, math.inf) if FP in taus.get(g, {}) else math.inf for g in F_grp], float)
    inreg = np.array([r in regions for r in F_reg]); kept = (F_conf >= tau) & inreg
    used = set(); per = {r: [0, 0, int(k)] for r, k in F_gt.items() if r in regions}
    for i in F_order:
        if not kept[i]:
            continue
        hit = False
        if i in F_pairs:
            best, bj = float("-inf"), -1
            for j, s_ in F_pairs[i]:
                if j not in used and s_ > best + 1e-9:
                    best, bj = s_, j
            if bj >= 0:
                used.add(bj); hit = True
        lst = per.setdefault(F_reg[i], [0, 0, 0]); lst[0 if hit else 1] += 1
    return per


def counts_x(taus, regions):
    n = counts_at(cV, taus, regions); n[FP] = finds_exact(taus, regions); return n


_chk = held = {cls: {} for cls in cB}
for fi in range(len(folds)):
    for cls, d in counts_at(cB, TB[fi], folds[fi]).items():
        held[cls].update(d)
_hx = {cls: {} for cls in cB}
for fi in range(len(folds)):
    _bb = {g: {c: v for c, v in d.items()} for g, d in TB[fi].items()}
    for cls, d in counts_x(_bb, folds[fi]).items():
        _hx[cls].update(d)
log(f"sanity exact-finds base (fgbm group absent) {M.score_counts(_hx)[0]:.7f}")
GRID = [round(0.05 * k, 2) for k in range(1, 20)]


def taus_of(fi, tau, rep):
    t = _copy.deepcopy(TB[fi]); t["maggr_fgbm"] = {FP: tau}
    if rep:
        t.setdefault("maggr", {})[FP] = math.inf
    return t


def held_score(tfun):
    held = {cls: {} for cls in cV}
    for fi in range(len(folds)):
        n = counts_x(tfun(fi), folds[fi])
        for cls, d in n.items():
            held[cls].update(d)
    return held


# sanity: UNI with tau=inf == base
h0 = held_score(lambda fi: taus_of(fi, math.inf, False)); s0 = M.score_counts(h0)[0]
log(f"sanity UNI tau=inf {s0:.7f}"); assert abs(s0 - T["V3P"]) < 1e-6
res = {"folds": T["folds"], "variants": {}, "gate_base": sB, "grid": GRID}
for name, rep in (("REP", True), ("UNI", False)):
    tr = {tau: [M.score_counts(counts_x(taus_of(fi, tau, rep), every - folds[fi]))[0] for fi in range(len(folds))] for tau in GRID}
    chosen = [max(GRID, key=lambda tau: (round(tr[tau][fi], 12), tau)) for fi in range(len(folds))]
    held = held_score(lambda fi: taus_of(fi, chosen[fi], rep))
    sg, per = M.score_counts(held)
    fixed_all = {tau: M.score_counts(held_score(lambda fi: taus_of(fi, tau, rep)))[0] for tau in GRID}
    log(f"{name} frozen: chosen {chosen} nested gate {sg:.6f} (base {sB:.6f}) finds F1 {per[FP]['f1']:.4f}")
    log(f"{name} fixed-tau gate (diag): " + " ".join(f"{t}:{v:.5f}" for t, v in fixed_all.items()))
    res["variants"][name] = dict(chosen=chosen, taus=[fin(taus_of(fi, chosen[fi], rep)) for fi in range(len(folds))],
                                 gate_nested=sg, finds_f1=per[FP]["f1"], extras="fg", train=tr, fixed_diag=fixed_all)
# unfrozen (secondary): full nested retune of all groups
eng_rep = _copy.copy(base3); eng_rep.base = dict(base3.base)
b = base3.base[FP]; keep = np.flatnonzero(b["grp"] != "maggr"); new = {int(o): n for n, o in enumerate(keep)}
eng_rep.base[FP] = dict(conf=b["conf"][keep], reg=b["reg"][keep], grp=b["grp"][keep], pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
for name, eng in (("REPu", eng_rep), ("UNIu", base3)):
    cu = curves_from(ex_v, eng=eng)
    n, ft, trs = run(cu)
    taus = [fin(t) for t in ft]
    if name == "REPu":
        for t in taus:
            t.setdefault("maggr", {})[FP] = None
    sg, per = M.score_counts(n)
    log(f"{name} unfrozen (approx greedy, secondary) nested gate {sg:.6f} finds F1 {per[FP]['f1']:.4f} fgbm taus {[t.get('maggr_fgbm') for t in taus]}")
    res["variants"][name] = dict(chosen=[None] * 10, taus=taus, gate_nested=sg, finds_f1=per[FP]["f1"], extras="fg")
res["taus_base"] = [fin(t) for t in ftB]
pickle.dump(dict(res=res, ex_base=ex_c19 + ex_dfx, ex_fg=ex_c19 + ex_dfx + ex_fg), open(OUT / "prep.pkl", "wb"))
json.dump(res, open(OUT / "prep_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
