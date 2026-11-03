#!/usr/bin/env python3
"""W1-SATQ frozen (PREREG docs/PREREG_W1_SATQ.md 0614536 + Amendment 1 docs/PREREG_W1_SATQ_AMEND1.md): gate-side nested
selection of Q1a/Q1b/Q1c, FP1/FP2, Q2 under the freeze rule (non-sp_dfine τ = base fold τ; sp_dfine per-class 1-D argmax).
Output /workspace/koz3/w1/satq/frzq/prep.pkl. Built from tools/satq/frz_prep.py (header + helpers)."""
import importlib.util, json, math, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]   # v3_gate_lib reads sys.argv
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D = G.D; M = D.M
_s = importlib.util.spec_from_file_location("sat_recall", "/workspace/koz3/wt/w1sq/expds/sat_recall.py")
SR = importlib.util.module_from_spec(_s); _s.loader.exec_module(SR)
import shapely
from shapely import wkb as WKB
from shapely.geometry import mapping, shape
from pyproj import Transformer

OUT = Path("/workspace/koz3/w1/satrec/out")
OUT2 = Path("/workspace/koz3/w1/satq/frzq"); OUT2.mkdir(parents=True, exist_ok=True)
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
folds = [set(f) for f in T["folds"]]; every = set().union(*folds); dreg = set(T["dreg"])
CL5 = D.CLASSES5
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
base3 = G.drop_engine(tab, lambda g, r: g == "etgr" or (g in G.SP and r in dreg)); del tab
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg2 = G.dfine_feats(); assert dreg2 == dreg
FB = json.load(open("/workspace/koz3/repo/docs/bots/2026-10-29/bot-10/work11/frame_boxes.json"))["boxes"]


def box_mpix(x0, y0, x1, y1):
    """W4 Q0 definition: (W·cos φ / 0.5)·(H·cos φ / 0.5)/1e6, φ = latitude of the box centre (EPSG:3857)."""
    c = math.cos(math.atan(math.sinh(((y0 + y1) / 2) / 6378137.0)))
    return (x1 - x0) * c / 0.5 * (y1 - y0) * c / 0.5 / 1e6


def drel_of(geom, reg):
    """(scene_drel = min over covering sp_or boxes, Mpix of the box giving that min, first on ties) or (None, None)."""
    b = FB.get(reg, {}).get("sp_or")
    if not b:
        return None, None
    pt = geom.representative_point()
    best, bm = None, None
    for x0, y0, x1, y1 in b:
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        if hw <= 0 or hh <= 0 or not (x0 <= pt.x <= x1 and y0 <= pt.y <= y1):
            continue
        d = max(abs(pt.x - (x0 + hw)) / hw, abs(pt.y - (y0 + hh)) / hh)
        if best is None or d < best:
            best, bm = d, box_mpix(x0, y0, x1, y1)
    return best, bm


def as_ex(f, dm=(None, None)):
    p = f["properties"]
    return dict(geometry=f["geometry"], class_name=p["class_name"], confidence=float(p["confidence"]), region_name=p["region_name"],
                markup_type=p.get("markup_type"), threshold_group=M.threshold_group(p), drel=dm[0], mpix=dm[1])


ex_c19 = [as_ex(f) for f in c19]
ex_dfx = [as_ex(f, drel_of(shape(f["geometry"]), f["properties"]["region_name"])) for f in dfx]
log("c19", len(ex_c19), "dfx", len(ex_dfx), "dfx drel None", sum(e["drel"] is None for e in ex_dfx))

# ---------- gate machinery ----------
def ex_of(lst):
    ex = base3._extra([{"type": "Feature", "geometry": e["geometry"],
                        "properties": {k: v for k, v in e.items() if k not in ("drel", "mpix", "geometry")}} for e in lst])
    assert sum(len(d["conf"]) for d in ex.values()) == len(lst), ("dropped extras", sum(len(d["conf"]) for d in ex.values()), len(lst))
    # attach drel in the same per-class order (_collect keeps input order of valid features)
    per = defaultdict(list)
    for e in lst:
        per[e["class_name"]].append(e)
    for c, d in ex.items():
        assert len(d["conf"]) == len(per[c]), (c, len(d["conf"]), len(per[c]))
        assert np.allclose(d["conf"], [e["confidence"] for e in per[c]])
        d["drel"] = np.array([np.nan if e["drel"] is None else e["drel"] for e in per[c]])
        d["mpix"] = np.array([np.nan if e["mpix"] is None else e["mpix"] for e in per[c]])
    return ex


def curves_from(ex, conf_fn=None, keep_fn=None):
    out = {}
    for c in base3.CL:
        b, e = base3.base[c], ex.get(c)
        nb = len(b["conf"])
        if e is not None:
            econf = e["conf"] if conf_fn is None else conf_fn(e, c)
            keep = np.ones(len(econf), bool) if keep_fn is None else keep_fn(e, econf, c)
            idx = np.flatnonzero(keep); rem = {int(o): n for n, o in enumerate(idx)}
            econf, ereg, egrp = econf[idx], e["reg"][idx], e["grp"][idx]
            epairs = {rem[i]: v for i, v in e["pairs"].items() if i in rem}
        conf = b["conf"] if e is None else np.concatenate([b["conf"], econf])
        reg = b["reg"] if e is None else np.concatenate([b["reg"], ereg])
        grp = b["grp"] if e is None else np.concatenate([b["grp"], egrp])
        pairs = dict(b["pairs"])
        if e is not None:
            pairs.update({nb + i: v for i, v in epairs.items()})
        order = np.argsort(-conf, kind="stable")
        rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
        hit = np.zeros(len(order), bool); used = set()
        for i in sorted(pairs, key=lambda i: rank[i]):
            best, bj = float("-inf"), -1
            for j, s_ in sorted(pairs[i]):
                if j not in used and s_ > best + 1e-9:
                    best, bj = s_, j
            if bj >= 0:
                used.add(bj); hit[rank[i]] = True
        out[c] = M.ClassCurve(conf=conf[order], hit=hit, pred_region=reg[order].tolist(), n_gt=len(base3.G[c]),
                              gt_regions=Counter(base3.greg[c]), pred_group=grp[order].tolist())
    return out


def counts_at(curves, taus, regions):
    counts = {cls: {} for cls in curves}
    for cls, c in curves.items():
        rnames, rcodes, gnames, gcodes = M._curve_codes(c)
        per = counts[cls]
        for r, k in c.gt_regions.items():
            if r in regions:
                per.setdefault(r, [0, 0, 0])[2] += int(k)
        lut = np.fromiter((r in regions for r in rnames), dtype=bool, count=len(rnames))
        m = lut[rcodes] if rcodes.size else np.zeros(0, bool)
        if not m.any():
            continue
        has = np.array([cls in taus.get(g, {}) for g in gnames], bool)
        tval = np.array([taus.get(g, {}).get(cls, 0.0) for g in gnames], np.float64)
        gh = gcodes[m]
        tau = np.where(~has[gh], np.inf, tval[gh])
        kept = c.conf[m] >= tau
        rk = rcodes[m][kept]; hk = c.hit[m][kept].astype(bool)
        tpc = np.bincount(rk[hk], minlength=len(rnames)); fpc = np.bincount(rk[~hk], minlength=len(rnames))
        for code in np.unique(rk).tolist():
            lst = per.setdefault(rnames[code], [0, 0, 0]); lst[0] += int(tpc[code]); lst[1] += int(fpc[code])
    return counts


def run(curves, fixed=frozenset()):
    n, _, ft = D.honest(curves, folds, fixed=fixed)
    tr = [M.score_counts(counts_at(curves, ft[fi], every - folds[fi]))[0] for fi in range(len(folds))]
    return n, ft, tr


fin = lambda t: {g: {c: (None if not math.isfinite(v) else float(v)) for c, v in d.items()} for g, d in t.items()}
res = {"folds": T["folds"], "variants": {}}

ex_base_l = ex_c19 + ex_dfx
ex_base = ex_of(ex_base_l)
cB = curves_from(ex_base)
nB, ftB, trB = run(cB)
sB = M.score_counts(nB)[0]
log(f"sanity gate base {sB:.7f} vs V3P {T['V3P']:.7f}")
assert abs(sB - T["V3P"]) < 1e-6, "gate base not reproduced"
res["gate_base"] = sB



# ======================= W1-SATQ =======================
import copy
ARKH = "arkhitektury"
FPC = ("kurgany_povrezhdennye", "kurgany_tselye", "arkhitektury", "fortifikatsii")   # v3_on frame_prior classes ∩ D-FINE
CLS = list(cB.keys()); assert ARKH in CLS and set(FPC) <= set(CLS), CLS
REGS = sorted(every); RI = {r: i for i, r in enumerate(REGS)}
W = np.array([M.CLASS_WEIGHTS.get(c, 0.0) for c in CLS]); CI = {c: i for i, c in enumerate(CLS)}


def held_arr(n):
    a = np.zeros((len(CLS), len(REGS), 3), np.int64)
    for ci, c in enumerate(CLS):
        for r, v in n.get(c, {}).items():
            a[ci, RI[r]] = v
    return a


def agg(counts, regions):
    a = np.zeros((len(CLS), 3))
    for ci, c in enumerate(CLS):
        for r, v in counts.get(c, {}).items():
            if r in regions:
                a[ci] += v
    return a


def f1(a):
    tp, fp, g = a[..., 0], a[..., 1], a[..., 2]
    den = 2 * tp + fp + (g - tp)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0)


def sc(t):
    f = f1(t); has = t[:, 2] > 0
    return float((W * f * has).sum() / (W * has).sum())


def hc_of(held):
    return {c: {REGS[ri]: held[ci, ri].tolist() for ri in range(len(REGS)) if held[ci, ri].any()} for ci, c in enumerate(CLS)}


def slices_delta(hc):
    return {nm: M.score_counts(hc, sl)[0] - M.score_counts(nB, sl)[0] for nm, sl in (("new34", set(T["new"])), ("nolidar", set(T["optical"])))}


def point_taus(curves, ftf, classes=None):
    """train [10,C,3] and held [C,R,3] counts at given per-fold taus (honest semantics: missing → drop)."""
    cur = curves if classes is None else {c: curves[c] for c in classes}
    tr = np.zeros((10, len(CLS), 3)); held = np.zeros((len(CLS), len(REGS), 3), np.int64)
    for fi in range(10):
        trs = every - folds[fi]
        tr[fi] = agg(counts_at(cur, ftf[fi], trs), trs)
        h = held_arr(counts_at(cur, ftf[fi], folds[fi]))
        idx = [RI[r] for r in folds[fi]]
        held[:, idx] = h[:, idx]
    return tr, held


# base point (raw taus) + sanity: counts_at at honest taus == honest held counts
_, _, ftB_raw = D.honest(cB, folds)
trB, heldB = point_taus(cB, ftB_raw)
assert np.array_equal(heldB, held_arr(nB)), "counts_at != honest held counts"
assert abs(M.score_counts(hc_of(heldB))[0] - sB) < 1e-12


def mult_prior(d0, s, classes):
    """m = max(0, 1 − s·clip((d − d0)/(1 − d0), 0, 1)) on sp_dfine of the given classes (frame_prior form: s = 1 − edge)."""
    def fn(e, c):
        if s <= 0 or c not in classes:
            return e["conf"]
        d = e["drel"]
        m = np.maximum(0.0, 1.0 - s * np.clip((np.nan_to_num(d, nan=0.0) - d0) / (1.0 - d0), 0.0, 1.0))
        off = np.isnan(d) | (e["grp"] != "sp_dfine")
        return e["conf"] * np.where(off, 1.0, m)
    return fn


def other_identical(tr, held, changed):
    oth = [CI[c] for c in CLS if c not in changed]
    return np.array_equal(held[oth], heldB[oth]) and np.allclose(tr[:, oth], trB[:, oth])


def assemble(choice_pts, name, chosen_specs):
    """choice_pts[fi] = (tr, held, ftf); returns variant dict."""
    held = heldB.copy(); taus = []
    for fi in range(10):
        _, h, ftf = choice_pts[fi]
        idx = [RI[r] for r in folds[fi]]
        held[:, idx] = h[:, idx]
        taus.append(fin(ftf[fi]))
    hc = hc_of(held)
    v = dict(chosen=chosen_specs, taus=taus, gate_nested=M.score_counts(hc)[0], hc=hc)
    v.update({"gate_d_" + k: x for k, x in slices_delta(hc).items()})
    log(name, "chosen", chosen_specs, f"nested gate {v['gate_nested']:.6f} Δ {v['gate_nested'] - sB:+.6f}",
        f"Δnew34 {v['gate_d_new34']:+.4f} Δnolidar {v['gate_d_nolidar']:+.4f}")
    return v


# ======================= W1-FRZ helpers =======================
SD = "sp_dfine"
DF_CLS = sorted({e["class_name"] for e in ex_dfx}); log("D-FINE classes", DF_CLS)


def mult_by_class(d0, s_of):
    """m_c = max(0, 1 − s_c·clip((d − d0)/(1 − d0), 0, 1)) on sp_dfine; s_of: class -> s."""
    def fn(e, c):
        s = s_of.get(c, 0.0)
        if s <= 0:
            return e["conf"]
        d = e["drel"]
        m = np.maximum(0.0, 1.0 - s * np.clip((np.nan_to_num(d, nan=0.0) - d0) / (1.0 - d0), 0.0, 1.0))
        off = np.isnan(d) | (e["grp"] != SD)
        return e["conf"] * np.where(off, 1.0, m)
    return fn


def frozen_taus(curves):
    """per fold: every non-sp_dfine group = base fold τ (ftB_raw); sp_dfine per-class τ = exact 1-D argmax of training F1."""
    out = []
    for fi in range(10):
        trs = every - folds[fi]
        t = {g: dict(d) for g, d in ftB_raw[fi].items() if g != SD}
        sd = {}
        for c, cv in curves.items():
            rnames, rcodes, gnames, gcodes = M._curve_codes(cv)
            lut = np.fromiter((r in trs for r in rnames), dtype=bool, count=len(rnames))
            m = lut[rcodes] if rcodes.size else np.zeros(0, bool)
            n_gt = int(sum(k for r, k in cv.gt_regions.items() if r in trs))
            if n_gt == 0 or not m.any():
                continue
            gn = np.asarray(gnames, dtype=object)[gcodes[m]]
            conf_, hit_ = cv.conf[m], cv.hit[m].astype(bool)
            isd = gn == SD
            if not isd.any():
                continue
            tv = np.array([t.get(g, {}).get(c, np.inf) for g in gn[~isd]], np.float64)
            tv = np.where(np.isfinite(tv), tv, np.inf)   # missing / None → dropped (honest semantics)
            ko = conf_[~isd] >= tv
            tp_o, n_o = float(hit_[~isd][ko].sum()), int(ko.sum())
            o = np.argsort(-conf_[isd], kind="stable")
            keep, tp, cut = M._group_cuts(conf_[isd][o], hit_[isd][o])
            tt, nn = tp_o + tp, n_o + keep
            f = np.where(nn > 0, M._f1(tt, nn - tt, n_gt - tt), 0.0)
            sd[c] = round(float(cut[int(np.argmax(f))]), 6)
        if sd:
            t[SD] = sd
        out.append(t)
    return out


# identity diagnostic
ftI = frozen_taus(cB)
mism = [(fi, c, ftB_raw[fi].get(SD, {}).get(c), ftI[fi].get(SD, {}).get(c)) for fi in range(10)
        for c in set(ftB_raw[fi].get(SD, {})) | set(ftI[fi].get(SD, {})) if ftB_raw[fi].get(SD, {}).get(c) != ftI[fi].get(SD, {}).get(c)]
trI, heldI = point_taus(cB, ftI)
log("identity frozen re-fit: sp_dfine τ mismatches vs base", len(mism), mism[:12],
    f"gate {M.score_counts(hc_of(heldI))[0]:.7f} (base {sB:.7f})")
res["identity_refit_mismatch"] = mism
for fi in range(10):   # freeze rule sanity: non-sp_dfine τ identical to base
    assert all(ftI[fi][g] == ftB_raw[fi][g] for g in ftB_raw[fi] if g != SD)

PTS = {}   # (d0, s) -> (tr, held, ftf) with s applied to ALL D-FINE classes; identity = base


def pt(d0, s):
    if s == 0:
        return (trB, heldB, ftB_raw)
    if (d0, s) not in PTS:
        cur = curves_from(ex_base, conf_fn=mult_by_class(d0, {c: s for c in DF_CLS}))
        ftf = frozen_taus(cur)
        tr, held = point_taus(cur, ftf)
        PTS[(d0, s)] = (tr, held, ftf)
        log("point", d0, s, f"nested-free honest-frozen gate {M.score_counts(hc_of(held))[0]:.6f}")
    return PTS[(d0, s)]


def combo(d0, s_of):
    """per-class separability: class c rows/τ from point (d0, s_of[c]) (frozen τ are per class)."""
    tr = np.zeros_like(trB); held = np.zeros_like(heldB); ftf = [{g: dict(d) for g, d in t.items() if g != SD} for t in ftB_raw]
    for c in CLS:
        s = s_of.get(c, 0.0) if c in DF_CLS else 0.0
        p = pt(d0, s); ci = CI[c]
        tr[:, ci] = p[0][:, ci]; held[ci] = p[1][ci]
        for fi in range(10):
            v = p[2][fi].get(SD, {}).get(c, "MISSING")
            if v != "MISSING":
                ftf[fi].setdefault(SD, {})[c] = v
    return tr, held, ftf


# exactness check of separability on one full point
_c = curves_from(ex_base, conf_fn=mult_by_class(0.25, {"kurgany_tselye": 0.5, **{c: 1.0 for c in DF_CLS if c != "kurgany_tselye"}}))
_f = frozen_taus(_c); _tr, _h = point_taus(_c, _f); _tr2, _h2, _f2 = combo(0.25, {"kurgany_tselye": 0.5, **{c: 1.0 for c in DF_CLS if c != "kurgany_tselye"}})
assert np.array_equal(_h, _h2) and np.allclose(_tr, _tr2) and all(_f[fi].get(SD) == _f2[fi].get(SD) for fi in range(10)), "separability"
log("separability check OK")


def select(cands, name, key_fn, spec_fn):
    """cands: list of (params, s_of, d0); key_fn(params) -> tie tuple (larger is better)."""
    pts, specs = [], []
    for fi in range(10):
        best = None
        for prm, d0, s_of in cands:
            tr = combo(d0, s_of)[0][fi]
            key = (round(sc(tr), 12),) + key_fn(prm)
            if best is None or key > best[0]:
                best = (key, (prm, d0, s_of))
        prm, d0, s_of = best[1]
        pts.append(combo(d0, s_of))
        sp = spec_fn(prm)
        if sp is not None:
            sp["train"] = best[0][0]
        specs.append(sp)
    return assemble(pts, name, specs)



D0S = (0.0, 0.25, 0.5); SS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5); TAUS = (0.15, 0.20, 0.25, 0.30, 0.373); ARKH = "arkhitektury"


def assert_freeze(name, v, changed):
    for fi in range(10):
        assert all(v["taus"][fi].get(g) == fin(ftB_raw[fi])[g] for g in ftB_raw[fi] if g != SD), ("freeze violated", name)
        for c in DF_CLS:
            if c not in changed:
                assert v["taus"][fi].get(SD, {}).get(c) == fin(ftB_raw[fi]).get(SD, {}).get(c), ("unchanged class τ moved", name, c)


# ---------------- Q1: arkh-only prior × recall τ (frozen) ----------------
Q1P = {}   # (d0, s, tau|None) -> (tr, held, ftf)
for d0 in D0S:
    for s in SS:
        if s == 0 and d0 != 0.0:
            continue
        cur = curves_from(ex_base, conf_fn=mult_by_class(d0, {ARKH: s}))
        ft = frozen_taus(cur)
        for tau in (None,) + TAUS:
            ftf = [{g: dict(d) for g, d in t.items()} for t in ft]
            if tau is not None:
                for t in ftf:
                    t.setdefault(SD, {})[ARKH] = tau
            tr, held = point_taus(cur, ftf)
            assert other_identical(tr, held, (ARKH,)), ("Q1 non-separable", d0, s, tau)
            Q1P[(d0, s, tau)] = (tr, held, ftf)
        log("Q1 point", d0, s, "frozen gate", f"{M.score_counts(hc_of(Q1P[(d0, s, None)][1]))[0]:.6f}")


def select_q1(keys, name):
    pts, specs = [], []
    for fi in range(10):
        best = None
        for (d0, s, tau) in keys:
            tr = Q1P[(d0, s, tau)][0][fi]
            key = (round(sc(tr), 12), -s, (tau if tau is not None else -1.0), d0)   # PREREG: smaller s, larger τ, larger d0
            if best is None or key > best[0]:
                best = (key, (d0, s, tau))
        d0, s, tau = best[1]
        pts.append(Q1P[(d0, s, tau)])
        specs.append(None if (s == 0 and tau is None) else dict(kind="q1", d0=d0, s_arkh=s, tau_arkh=tau, train=best[0][0]))
    return assemble(pts, name, specs)


K = list(Q1P)
res["variants"]["Q1a"] = select_q1([k for k in K if k[2] is None], "Q1a")
res["variants"]["Q1b"] = select_q1([k for k in K if k[2] is not None] + [(0.0, 0.0, None)], "Q1b")
res["variants"]["Q1c"] = select_q1([k for k in K if k[1] == 0 and k[2] is not None], "Q1c")
for nm in ("Q1a", "Q1b", "Q1c"):
    assert_freeze(nm, res["variants"][nm], (ARKH,))

# ---------------- FP: frame_prior rebound to sp_dfine (frozen) ----------------
FPP = {}
def fp_point(d0, edge):
    if (d0, edge) not in FPP:
        if edge >= 1.0:
            FPP[(d0, edge)] = (trB, heldB, ftB_raw)
        else:
            cur = curves_from(ex_base, conf_fn=mult_by_class(d0, {c: 1.0 - edge for c in FPC}))
            ft = frozen_taus(cur)
            tr, held = point_taus(cur, ft)
            assert other_identical(tr, held, FPC), ("FP non-separable", d0, edge)
            FPP[(d0, edge)] = (tr, held, ft)
        log("FP point", d0, edge, f"frozen gate {M.score_counts(hc_of(FPP[(d0, edge)][1]))[0]:.6f}")
    return FPP[(d0, edge)]


res["variants"]["FP1"] = assemble([fp_point(0.5, 0.5)] * 10, "FP1", [dict(kind="fp", d0=0.5, edge=0.5)] * 10)
FD0 = (0.25, 0.5, 0.75); FE = (0.0, 0.25, 0.5, 0.75, 1.0)
pts, specs = [], []
for fi in range(10):
    best = None
    for d0 in FD0:
        for e in FE:
            if e >= 1.0 and d0 != FD0[-1]:
                continue   # identity once
            tr = fp_point(d0, e)[0][fi]
            key = (round(sc(tr), 12), e, d0)   # PREREG: larger edge, then larger d0
            if best is None or key > best[0]:
                best = (key, (d0, e))
    d0, e = best[1]
    pts.append(fp_point(d0, e)); specs.append(None if e >= 1.0 else dict(kind="fp", d0=d0, edge=e, train=best[0][0]))
res["variants"]["FP2"] = assemble(pts, "FP2", specs)
for nm in ("FP1", "FP2"):
    assert_freeze(nm, res["variants"][nm], FPC)

# ---------------- Q2: framing gate on scenes >= 4 Mpix (Amendment 1 §C) ----------------
Q2C = ("kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury"); A_MPIX = 4.0
XS = (0.4, 0.5, 0.6); T2S = (0.35, 0.5, 0.7, 0.85)
assert set(Q2C) <= set(DF_CLS), DF_CLS
n_big = sum(1 for e in ex_dfx if e["mpix"] is not None and e["mpix"] >= A_MPIX)
log("Q2: sp_dfine feats on scenes >= 4 Mpix", n_big, "of", len(ex_dfx))
res["q2_n_big"] = n_big


def q2_keep(x, t2):
    def fn(e, econf, c):
        if c not in Q2C:
            return np.ones(len(econf), bool)
        d = np.nan_to_num(e["drel"], nan=-1.0); mp = np.nan_to_num(e["mpix"], nan=-1.0)
        drop = (e["grp"] == SD) & (mp >= A_MPIX) & (d > x) & (econf < t2)
        return ~drop
    return fn


Q2P = {}
def q2_point(x, t2):
    if (x, t2) not in Q2P:
        if x is None:
            Q2P[(x, t2)] = (trB, heldB, ftB_raw)
        else:
            cur = curves_from(ex_base, keep_fn=q2_keep(x, t2))
            ft = frozen_taus(cur)
            tr, held = point_taus(cur, ft)
            assert other_identical(tr, held, Q2C), ("Q2 non-separable", x, t2)
            Q2P[(x, t2)] = (tr, held, ft)
            log("Q2 point", x, t2, f"frozen gate {M.score_counts(hc_of(held))[0]:.6f}")
    return Q2P[(x, t2)]


cands = [(None, 0.0)] + [(x, t2) for x in XS for t2 in T2S]
pts, specs = [], []
for fi in range(10):
    best = None
    for x, t2 in cands:
        tr = q2_point(x, t2)[0][fi]
        key = (round(sc(tr), 12), -t2, x if x is not None else 9.0)   # Amendment: identity, then smaller t2, then larger x
        if best is None or key > best[0]:
            best = (key, (x, t2))
    x, t2 = best[1]
    pts.append(q2_point(x, t2)); specs.append(None if x is None else dict(kind="q2", x=x, t2=t2, min_scene_mpix=A_MPIX, train=best[0][0]))
res["variants"]["Q2"] = assemble(pts, "Q2", specs)
assert_freeze("Q2", res["variants"]["Q2"], Q2C)

res["Q1_points"] = {f"{k[0]}_{k[1]}_{k[2]}": M.score_counts(hc_of(v[1]))[0] for k, v in Q1P.items()}
res["FP_points"] = {f"{k[0]}_{k[1]}": M.score_counts(hc_of(v[1]))[0] for k, v in FPP.items()}
res["Q2_points"] = {f"{k[0]}_{k[1]}": M.score_counts(hc_of(v[1]))[0] for k, v in Q2P.items()}
res["taus_base"] = [fin(t) for t in ftB]
res["nB"] = nB
pickle.dump(dict(res=res, ex_base=ex_base_l), open(OUT2 / "prep.pkl", "wb"))
summ = {k: v for k, v in res.items() if k not in ("nB", "variants")}
summ["variants"] = {k: {a: b for a, b in v.items() if a not in ("extra", "hc")} for k, v in res["variants"].items()}
json.dump(summ, open(OUT2 / "prep_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
