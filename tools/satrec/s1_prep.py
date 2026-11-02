#!/usr/bin/env python3
"""W1-SATREC step 1 (PREREG docs/PREREG_W1_SATREC.md 140839a): gate-side nested selection for R1/R2/R3/C1/C2.
Output: /workspace/koz3/w1/satrec/out/prep.pkl = per variant: per-fold taus + per-fold hyperparams + extras list."""
import importlib.util, json, math, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D = G.D; M = D.M
_s = importlib.util.spec_from_file_location("sat_recall", "/workspace/koz3/wt/w1sr/expds/sat_recall.py")
SR = importlib.util.module_from_spec(_s); _s.loader.exec_module(SR)
import shapely
from shapely import wkb as WKB
from shapely.geometry import mapping, shape
from pyproj import Transformer

OUT = Path("/workspace/koz3/w1/satrec/out")
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

# ---------- fusion members (r18 + f2 from the table, UTM -> 3857) + dfx ----------
tb = pickle.load(open("/workspace/koz3/w5data/work/table.pkl", "rb"))
CLt = list(tb["CL"]); tcls = np.asarray(tb["cls"]); tconf = np.asarray(tb["conf"], float); treg = np.asarray(tb["reg"]).astype(str)
tgrp = np.asarray(tb["grp"]).astype(str); del tb
GW = pickle.load(open("/workspace/koz3/w5data/work/geom_wkb.pkl", "rb"))
msk = np.isin(tgrp, list(G.SP)); assert set(treg[msk]) <= dreg, "sp rows outside dreg"
zone = {}
for e in ex_dfx:
    zone.setdefault(e["region_name"], []).append(shape(e["geometry"]).centroid.x)
to4326 = Transformer.from_crs(3857, 4326, always_xy=True)
for r, xs in zone.items():
    lon, _ = to4326.transform(float(np.median(xs)), 0.0)
    zone[r] = 32600 + int((lon + 180) // 6) + 1
mem = []
bad = 0
TR = {}
for i in np.flatnonzero(msk):
    r = treg[i]; ep = zone[r]
    tr = TR.setdefault(ep, Transformer.from_crs(ep, 3857, always_xy=True))
    g = shapely.transform(WKB.loads(GW[i]), lambda xy: np.c_[tr.transform(xy[:, 0], xy[:, 1])])
    mem.append(dict(geom=g, class_name=CLt[tcls[i]], confidence=float(tconf[i]), region_name=r, markup_type="sp_or", threshold_group=tgrp[i]))
del GW
# CRS sanity: member centroids inside the region's sp_or scenes
inside = sum(drel_of(m["geom"], m["region_name"]) is not None for m in mem)
log("members from table", len(mem), Counter(m["threshold_group"] for m in mem), "inside scenes", inside)
for e in ex_dfx:
    mem.append(dict(geom=shape(e["geometry"]), class_name=e["class_name"], confidence=e["confidence"], region_name=e["region_name"],
                    markup_type="sp_or", threshold_group="sp_dfine"))
fused = SR.sat_equal_fusion(mem, {"radius_m": 8.0})
assert all(f["threshold_group"] == "sp_fuse" for f in fused)
ex_fuse = [dict(geometry=mapping(f["geom"]), class_name=f["class_name"], confidence=f["confidence"], region_name=f["region_name"],
                markup_type="sp_or", threshold_group="sp_fuse", drel=drel_of(f["geom"], f["region_name"])) for f in fused]
log("fused", len(ex_fuse), Counter(e["class_name"] for e in ex_fuse), "anchors", Counter(f["fuse_anchor"] for f in fused))
del mem, fused


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
            econf = e["conf"] if conf_fn is None else conf_fn(e)
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

ex_base = ex_of(ex_c19 + ex_dfx)
cB = curves_from(ex_base)
nB, ftB, trB = run(cB)
sB = M.score_counts(nB)[0]
log(f"sanity gate base {sB:.7f} vs V3P {T['V3P']:.7f}")
assert abs(sB - T["V3P"]) < 1e-6, "gate base not reproduced"
res["gate_base"] = sB


def nested(name, grid, make, fixed_fn=lambda p: frozenset(), post_taus=None, tie=None):
    rows = []
    for p in grid:
        cv = make(p)
        n, ft, tr = run(cv, fixed_fn(p))
        rows.append((p, n, ft, tr))
        log(f"{name} {p}: gate honest {M.score_counts(n)[0]:.6f} train mean {np.mean(tr):.6f}")
    chosen, taus, held = [], [], {cls: {} for cls in cB}
    for fi in range(len(folds)):
        best = max(range(len(rows)), key=lambda k: (round(rows[k][3][fi], 12), tie(rows[k][0]) if tie else 0))
        p, n, ft, _ = rows[best]
        t = fin(ft[fi])
        if post_taus:
            t = post_taus(p, t)
        chosen.append(p); taus.append(t)
        for cls, d in n.items():
            for r, v in d.items():
                if r in folds[fi]:
                    held[cls][r] = v
    sg = M.score_counts(held)[0]
    log(f"{name}: chosen {chosen} nested gate {sg:.6f} (base gate {sB:.6f})")
    return dict(chosen=chosen, taus=taus, gate_nested=sg, grid=[(p, M.score_counts(n)[0], list(tr)) for p, n, _, tr in rows])


DF_CL = list(CL5)
# R1: single tau for sp_dfine (prefilter + fixed)
r1 = nested("R1", [round(0.20 + 0.05 * k, 2) for k in range(11)],
            lambda tau: curves_from(ex_base, keep_fn=lambda e, cf: (e["grp"] != "sp_dfine") | (cf >= tau)),
            fixed_fn=lambda p: frozenset({"sp_dfine"}),
            post_taus=lambda tau, t: {**t, "sp_dfine": {c: tau for c in DF_CL}}, tie=lambda tau: -tau)
res["variants"]["R1"] = dict(r1, extras="base")

ex_fz = ex_of(ex_c19 + ex_fuse)
r2c = curves_from(ex_fz); n2, ft2, tr2 = run(r2c)
log(f"R2 gate honest {M.score_counts(n2)[0]:.6f}")
res["variants"]["R2"] = dict(chosen=[None] * 10, taus=[fin(t) for t in ft2], gate_nested=M.score_counts(n2)[0], extras="fuse")
r3 = nested("R3", [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
            lambda tau: curves_from(ex_fz, keep_fn=lambda e, cf: (e["grp"] != "sp_fuse") | (cf >= tau)),
            fixed_fn=lambda p: frozenset({"sp_fuse"}),
            post_taus=lambda tau, t: {**t, "sp_fuse": {c: tau for c in DF_CL}}, tie=lambda tau: -tau)
res["variants"]["R3"] = dict(r3, extras="fuse")

CGRID = [(0.0, 0.0)] + [(d0, s) for d0 in (0.0, 0.25, 0.5, 0.75) for s in (0.25, 0.5, 0.75)]


def mult(e, grp, d0, s):
    d = e["drel"]
    m = 1.0 - s * np.clip((np.nan_to_num(d, nan=0.0) - d0) / (1.0 - d0), 0.0, 1.0)
    m = np.where(np.isnan(d) | (e["grp"] != grp), 1.0, m)
    return e["conf"] * m


c1 = nested("C1", CGRID, lambda p: curves_from(ex_base, conf_fn=lambda e: mult(e, "sp_dfine", *p)), tie=lambda p: (-p[1], p[0]))
res["variants"]["C1"] = dict(c1, extras="base", prior_group="sp_dfine")
c2 = nested("C2", CGRID, lambda p: curves_from(ex_fz, conf_fn=lambda e: mult(e, "sp_fuse", *p)), tie=lambda p: (-p[1], p[0]))
res["variants"]["C2"] = dict(c2, extras="fuse", prior_group="sp_fuse")
res["taus_base"] = [fin(t) for t in ftB]
pickle.dump(dict(res=res, ex_base=ex_c19 + ex_dfx, ex_fuse=ex_c19 + ex_fuse), open(OUT / "prep.pkl", "wb"))
json.dump({k: v for k, v in res.items()}, open(OUT / "prep_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
log("done")
