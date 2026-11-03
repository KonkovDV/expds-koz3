#!/usr/bin/env python3
"""W1-FRZ full-data selection (handoff, docs/W1_C1F_HANDOFF.md): C1F / P2F parameters + sp_dfine τ chosen ONCE on all regions,
non-sp_dfine τ frozen at the full-data base sweep; same selection rules as PREREG FRZ 1532475. Output /workspace/koz3/w1/satq/frz/full.pkl.
Gate machinery = tools/satdet/t1_prep.py (W1-SATDET)."""
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
OUT2 = Path("/workspace/koz3/w1/satq/frz"); OUT2.mkdir(parents=True, exist_ok=True)
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


def drel_of(geom, reg):
    b = FB.get(reg, {}).get("sp_or")
    if not b:
        return None
    pt = geom.representative_point()
    best = None
    for x0, y0, x1, y1 in b:
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        if hw <= 0 or hh <= 0 or not (x0 <= pt.x <= x1 and y0 <= pt.y <= y1):
            continue
        d = max(abs(pt.x - (x0 + hw)) / hw, abs(pt.y - (y0 + hh)) / hh)
        best = d if best is None else min(best, d)
    return best


def as_ex(f, drel=None):
    p = f["properties"]
    return dict(geometry=f["geometry"], class_name=p["class_name"], confidence=float(p["confidence"]), region_name=p["region_name"],
                markup_type=p.get("markup_type"), threshold_group=M.threshold_group(p), drel=drel)


ex_c19 = [as_ex(f) for f in c19]
ex_dfx = [as_ex(f, drel_of(shape(f["geometry"]), f["properties"]["region_name"])) for f in dfx]
log("c19", len(ex_c19), "dfx", len(ex_dfx), "dfx drel None", sum(e["drel"] is None for e in ex_dfx))

# ---------- gate machinery ----------
def ex_of(lst):
    ex = base3._extra([{"type": "Feature", "geometry": e["geometry"],
                        "properties": {k: v for k, v in e.items() if k not in ("drel", "geometry")}} for e in lst])
    assert sum(len(d["conf"]) for d in ex.values()) == len(lst), ("dropped extras", sum(len(d["conf"]) for d in ex.values()), len(lst))
    # attach drel in the same per-class order (_collect keeps input order of valid features)
    per = defaultdict(list)
    for e in lst:
        per[e["class_name"]].append(e)
    for c, d in ex.items():
        assert len(d["conf"]) == len(per[c]), (c, len(d["conf"]), len(per[c]))
        assert np.allclose(d["conf"], [e["confidence"] for e in per[c]])
        d["drel"] = np.array([np.nan if e["drel"] is None else e["drel"] for e in per[c]])
    return ex


def curves_from(ex, conf_fn=None, keep_fn=None):
    out = {}
    for c in base3.CL:
        b, e = base3.base[c], ex.get(c)
        nb = len(b["conf"])
        if e is not None:
            econf = e["conf"] if conf_fn is None else conf_fn(e, c)
            keep = np.ones(len(econf), bool) if keep_fn is None else keep_fn(e, econf)
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


# ======================= W1-FRZ =======================
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



# ======================= full-data selection =======================
EV = set(every)


def sweep_all(curves):
    """base sweep on ALL regions (D.honest semantics with train = every)."""
    taus = {}
    for c, cv in curves.items():
        rnames, rcodes, gnames, gcodes = M._curve_codes(cv)
        n_gt = int(sum(cv.gt_regions.values()))
        r = D.sweep_fixed(cv.conf, cv.hit, gcodes, gnames, n_gt, 20, frozenset())
        if r is not None:
            for g, gv in r["groups"].items():
                taus.setdefault(g, {})[c] = gv["threshold"]
    return taus


def frozen_full(curves, base):
    t = {g: dict(d) for g, d in base.items() if g != SD}; sd = {}
    for c, cv in curves.items():
        rnames, rcodes, gnames, gcodes = M._curve_codes(cv)
        n_gt = int(sum(cv.gt_regions.values()))
        if n_gt == 0 or not rcodes.size:
            continue
        gn = np.asarray(gnames, dtype=object)[gcodes]; isd = gn == SD
        if not isd.any():
            continue
        conf_, hit_ = cv.conf, cv.hit.astype(bool)
        tv = np.array([t.get(g, {}).get(c, np.inf) for g in gn[~isd]], np.float64); tv = np.where(np.isfinite(tv), tv, np.inf)
        ko = conf_[~isd] >= tv; tp_o, n_o = float(hit_[~isd][ko].sum()), int(ko.sum())
        o = np.argsort(-conf_[isd], kind="stable")
        keep, tp, cut = M._group_cuts(conf_[isd][o], hit_[isd][o])
        tt, nn = tp_o + tp, n_o + keep
        f = np.where(nn > 0, M._f1(tt, nn - tt, n_gt - tt), 0.0)
        sd[c] = round(float(cut[int(np.argmax(f))]), 6)
    t[SD] = sd
    return t


BT = sweep_all(cB)
rowsB = agg(counts_at(cB, BT, EV), EV); sBF = sc(rowsB)
assert frozen_full(cB, BT)[SD] == BT[SD], ("identity full re-fit != base sweep", frozen_full(cB, BT)[SD], BT[SD])
log(f"full-data base in-sample score {sBF:.6f}; base sp_dfine τ {BT[SD]}")
FP_ = {}


def fpt(d0, s):
    if s == 0:
        return rowsB, BT
    if (d0, s) not in FP_:
        cur = curves_from(ex_base, conf_fn=mult_by_class(d0, {c: s for c in DF_CLS}))
        t = frozen_full(cur, BT)
        FP_[(d0, s)] = (agg(counts_at(cur, t, EV), EV), t)
        log("full point", d0, s, f"in-sample {sc(FP_[(d0, s)][0]):.6f}")
    return FP_[(d0, s)]


def fcombo(d0, s_of):
    rows = np.zeros_like(rowsB); t = {g: dict(d) for g, d in BT.items() if g != SD}; t[SD] = {}
    for c in CLS:
        s = s_of.get(c, 0.0) if c in DF_CLS else 0.0
        r, tt = fpt(d0, s); rows[CI[c]] = r[CI[c]]
        if c in tt.get(SD, {}):
            t[SD][c] = tt[SD][c]
    return rows, t


def fsel(cands, key_fn):
    best = None
    for prm, d0, s_of in cands:
        key = (round(sc(fcombo(d0, s_of)[0]), 12),) + key_fn(prm)
        if best is None or key > best[0]:
            best = (key, (prm, d0, s_of))
    return best


SS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5); GOR = "gorodishcha"; ARKH = "arkhitektury"
out = {"full_base_insample": sBF, "BT": fin(BT)}
bC = fsel([((0.0, 0.0), 0.0, {})] + [((d0, s), d0, {c: s for c in DF_CLS}) for d0 in (0.0, 0.25, 0.5, 0.75) for s in (0.25, 0.5, 0.75)],
          lambda p: (-p[1], p[0]))
bP = fsel([((d0, sg, sa), d0, {GOR: sg, ARKH: sa}) for d0 in (0.0, 0.25, 0.5) for sg in SS for sa in SS], lambda p: (-p[1], -p[2], p[0]))
for nm, b in (("C1F", bC), ("P2F", bP)):
    prm, d0, s_of = b[1]; rows, t = fcombo(d0, s_of)
    dcls = {c: float(f1(rows[CI[c]]) - f1(rowsB[CI[c]])) for c in CLS}
    out[nm] = dict(d0=d0, s_by_class={c: float(s_of[c]) for c in s_of}, sp_dfine_tau=t[SD], insample=sc(rows),
                   d_insample=sc(rows) - sBF, d_cls=dcls, all_cls_ge0=all(v >= -1e-12 for v in dcls.values()), taus=fin(t))
    log(nm, "full-data choice d0", d0, "s", s_of, "τ", t[SD], f"in-sample {sc(rows):.6f} Δ {sc(rows) - sBF:+.6f}", "Δcls", {c: round(v, 4) for c, v in dcls.items()})

# sensitivity only (selection above unchanged): s beyond the grid
sens = {}
for d0 in (0.0, 0.25, 0.5, 0.75):
    for s in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        sens[f"C1_all d0={d0} s={s}"] = sc(fpt(d0, s)[0])
SX = SS + (1.75, 2.0)
bPx = fsel([((d0, sg, sa), d0, {GOR: sg, ARKH: sa}) for d0 in (0.0, 0.25, 0.5) for sg in SX for sa in SX], lambda p: (-p[1], -p[2], p[0]))
sens["P2F_ext_best"] = dict(prm=bPx[1][0], insample=bPx[0][0])
sens["C1F_ext_best"] = dict(zip(("prm", "insample"), (lambda b: (b[1][0], b[0][0]))(
    fsel([((d0, s), d0, {c: s for c in DF_CLS}) for d0 in (0.0, 0.25, 0.5, 0.75) for s in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)], lambda p: (-p[1], p[0])))))
# nested honest-frozen (no selection) for C1-all points beyond the grid, for context
for d0 in (0.0, 0.25):
    for s in (1.0, 1.5, 2.0):
        cur = curves_from(ex_base, conf_fn=mult_by_class(d0, {c: s for c in DF_CLS})); ftf = frozen_taus(cur); _, h = point_taus(cur, ftf)
        sens[f"nested C1_all d0={d0} s={s}"] = M.score_counts(hc_of(h))[0]
for d0 in (0.0, 0.25):
    for sg, sa in ((1.5, 1.5), (1.75, 1.75), (2.0, 2.0), (2.0, 1.5)):
        cur = curves_from(ex_base, conf_fn=mult_by_class(d0, {GOR: sg, ARKH: sa})); ftf = frozen_taus(cur); _, h = point_taus(cur, ftf)
        sens[f"nested P2 d0={d0} sg={sg} sa={sa}"] = M.score_counts(hc_of(h))[0]
out["sensitivity"] = sens
log("sensitivity", json.dumps(sens, default=str))
# replay inputs: same taus/spec in every fold (in-sample full-data config)
spec = lambda nm: dict(kind="pr", d0=out[nm]["d0"], s=0.0, s_by_class=out[nm]["s_by_class"])
V = {"B_full": dict(taus=[fin(BT)] * 10, chosen=[None] * 10),
     "C1F_full": dict(taus=[out["C1F"]["taus"]] * 10, chosen=[spec("C1F")] * 10),
     "P2F_full": dict(taus=[out["P2F"]["taus"]] * 10, chosen=[spec("P2F")] * 10)}
pickle.dump(dict(res=dict(variants=V, taus_base=res.get("taus_base")), ex_base=ex_base_l), open(OUT2 / "full.pkl", "wb"))
json.dump(out, open(OUT2 / "full_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
