#!/usr/bin/env python3
"""LI-KURG-TAU gate (prereg/LI_KURG_TAU_PREREG.md). Clean-room. V3P context as kdual.py."""
import importlib.util, json, sys, gc, pickle
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, log, site_family = G.D, G.M, G.log, G.site_family
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = Path("/workspace/koz3/orch/bot-W3/runs/gate_LIKTF.json")
PO, TS = "kurgany_povrezhdennye", "kurgany_tselye"; KC = (PO, TS); LIG = {"li", "li_dev", "li_mosaic"}
R_CONF, C_MIN, PMIN = 15.0, 0.10, 1 - 0.1 / 4
res = {"boot": BOOT, "sanity": {}, "variants": {}}
def save(): OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
TP = Path("/workspace/koz3/w5data/work/table.pkl")
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(TP, stk)
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10); del cA; gc.collect()
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg = G.dfine_feats(); SP = G.SP
drop = lambda g, r: g == "etgr" or (g in SP and r in dreg)
b3 = G.drop_engine(tab, drop)
cP, _ = b3.curves(c19 + dfx)
nB, _, tB = D.honest(cP, folds)
sB = M.score_counts(nB)[0]; res["sanity"]["V3P"] = sB; log(f"V3P honest {sB:.6f}"); save()
ex = b3._extra(c19 + dfx)
pr = M._collect(b3.mod, b3.calc, D.S.fc(c19 + dfx), False); M._UTM_GEOM_CACHE.clear()
R = {}   # per kurgan class: row arrays base+extra
for c in KC:
    b, e = b3.base[c], ex.get(c); nb = len(b["conf"])
    conf = b["conf"] if e is None else np.concatenate([b["conf"], e["conf"]])
    reg = b["reg"] if e is None else np.concatenate([b["reg"], e["reg"]])
    grp = b["grp"] if e is None else np.concatenate([b["grp"], e["grp"]])
    pairs = dict(b["pairs"])
    if e is not None: pairs.update({nb + i: v for i, v in e["pairs"].items()})
    xy = np.full((len(conf), 2), np.nan)
    if e is not None:
        lst = pr.get(c, []); assert len(lst) == len(e["conf"]) and np.allclose([x[1] for x in lst], e["conf"])
        for i, x in enumerate(lst): ct = x[0].centroid; xy[nb + i] = (ct.x, ct.y)
    R[c] = dict(conf=conf.astype(np.float64), reg=reg.astype(str), grp=grp.astype(str), pairs=pairs, nb=nb, xy=xy)
del pr, ex; gc.collect()
with open(TP, "rb") as fh: T = pickle.load(fh)
for k, c in enumerate(T["CL"]):
    if c not in KC: continue
    idx = np.flatnonzero(T["cls"] == k)
    keep = np.array([not drop(g, r) for g, r in zip(tab.base[c]["grp"], tab.base[c]["reg"])], bool)
    gi = idx[keep]; assert len(gi) == R[c]["nb"]
    R[c]["xy"][:R[c]["nb"]] = np.c_[T["cx"][gi], T["cy"][gi]]
    R[c]["key"] = [(T["reg"][i], round(float(T["cx"][i]), 3), round(float(T["cy"][i]), 3), round(float(T["area"][i]), 3), int(T["nv"][i]), str(T["grp"][i])) for i in gi]
del T, tab; gc.collect()
for c in KC: R[c]["li"] = np.isin(R[c]["grp"], list(LIG))
def curve(c, conf, grp):
    A = R[c]; order = np.argsort(-conf, kind="stable"); rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    hit = np.zeros(len(order), bool); used = set()
    for i in sorted(A["pairs"], key=lambda i: rank[i]):
        best, bj = float("-inf"), -1
        for j, s_ in sorted(A["pairs"][i]):
            if j not in used and s_ > best + 1e-9: best, bj = s_, j
        if bj >= 0: used.add(bj); hit[rank[i]] = True
    return M.ClassCurve(conf=conf[order], hit=hit, pred_region=A["reg"][order].tolist(), n_gt=cP[c].n_gt, gt_regions=cP[c].gt_regions, pred_group=grp[order].tolist())
for c in KC:
    cv = curve(c, R[c]["conf"], R[c]["grp"])
    assert np.array_equal(cv.conf, cP[c].conf) and np.array_equal(cv.hit, cP[c].hit) and cv.pred_group == cP[c].pred_group
res["sanity"]["rebuild_equal"] = True; log("rebuild == V3P"); save()
# variant scores
def sA(c):
    o = KC[1 - KC.index(c)]; om = {k: R[o]["conf"][p] for p, k in enumerate(R[o]["key"]) if R[o]["li"][p]}
    s = R[c]["conf"].copy()
    for p in np.flatnonzero(R[c]["li"][:R[c]["nb"]]): s[p] += om.get(R[c]["key"][p], 0.0)
    return s, R[c]["grp"]
def sB(c, form):
    A = R[c]; s = A["conf"].copy(); li = np.flatnonzero(A["li"]); by = defaultdict(list)
    for p in li: by[A["reg"][p]].append(p)
    for r, ps in by.items():
        ps = np.array(ps); o = ps[np.argsort(-A["conf"][ps], kind="stable")]; n = len(o); rk = np.arange(n)
        s[o] = 1 - rk / n if form == "q" else 1.0 / (1 + rk)
    return s, A["grp"]
def sC(c):
    ok = {}
    for k in KC:
        A = R[k]; m = (~A["li"]) & (A["conf"] >= C_MIN) & np.isfinite(A["xy"][:, 0])
        for r in np.unique(A["reg"][m]): ok.setdefault(r, []).append(A["xy"][m & (A["reg"] == r)])
    tr = {r: cKDTree(np.vstack(v)) for r, v in ok.items()}
    A = R[c]; grp = A["grp"].astype(object).copy(); n = 0
    for r, t in tr.items():
        ps = np.flatnonzero(A["li"] & (A["reg"] == r))
        if not len(ps): continue
        d, _ = t.query(A["xy"][ps], k=1); cf = ps[d <= R_CONF]; n += len(cf)
        grp[cf] = np.array([g + "|x" for g in A["grp"][cf]], dtype=object)
    res["sanity"][f"C_confirmed_{c}"] = [n, int(A["li"].sum())]
    return A["conf"], grp.astype(str)
VAR = {"A": lambda c: sA(c), "B1": lambda c: sB(c, "q"), "B2": lambda c: sB(c, "k"), "C": sC}

import copy
inl = lambda g: g.split("|")[0] in LIG
every = set().union(*folds)
def frozen_taus(CV):
    out = []
    for k, held in enumerate(folds):
        train = every - held; tk = copy.deepcopy(tB[k])
        for g in list(tk):
            if inl(g):
                for c in KC: tk[g].pop(c, None)
        for c in KC:
            cv = CV[c]; tr = np.fromiter((r in train for r in cv.pred_region), bool, len(cv.conf))
            n_gt = int(sum(v for r, v in cv.gt_regions.items() if r in train))
            grp = np.asarray(cv.pred_group).astype(str); conf = np.asarray(cv.conf); hit = np.asarray(cv.hit).astype(bool)
            lay = np.array([inl(g) for g in grp], bool)
            ft = np.array([tB[k].get(g, {}).get(c, np.inf) for g in grp]) if len(grp) else np.zeros(0)
            fm = tr & ~lay & (conf >= ft); tp0, n0 = int(hit[fm].sum()), int(fm.sum())
            groups = sorted(set(grp[tr & lay]))
            if n_gt == 0 or not groups: continue
            cuts = {}
            for g in groups:
                m = tr & lay & (grp == g); o = np.argsort(-conf[m], kind="stable"); cs, hs = conf[m][o], hit[m][o]
                cuts[g] = M._group_cuts(cs, hs)
            f1 = lambda tp, n: float(M._f1(tp, n - tp, n_gt - tp)) if n else 0.0
            best, bf = None, -1.0
            for start in ("none", "all"):
                kk = {g: 0 if start == "none" else len(cuts[g][0]) - 1 for g in groups}
                for _ in range(20):
                    ch = False
                    for g in groups:
                        tpo = tp0 + sum(cuts[h][1][kk[h]] for h in groups if h != g); no = n0 + sum(cuts[h][0][kk[h]] for h in groups if h != g)
                        f = np.array([f1(tpo + cuts[g][1][j], no + cuts[g][0][j]) for j in range(len(cuts[g][0]))])
                        j = int(np.argmax(f))
                        if j != kk[g] and f[j] > f[kk[g]] + 1e-12: kk[g], ch = j, True
                    if not ch: break
                f = f1(tp0 + sum(cuts[g][1][kk[g]] for g in groups), n0 + sum(cuts[g][0][kk[g]] for g in groups))
                if f > bf: bf, best = f, dict(kk)
            for g in groups: tk.setdefault(g, {})[c] = round(float(cuts[g][2][best[g]]), 6)
        out.append(tk)
    return out
fz = frozen_taus(cP); nS, _, _ = D.honest(cP, folds, frozen=fz); sS = M.score_counts(nS)[0]
res["sanity"]["frozen_on_V3P"] = sS; log(f"sanity frozen sweep on V3P {sS:.6f} vs {res['sanity']['V3P']:.6f}"); save()
if len(sys.argv) > 2 and sys.argv[2] == "sanity": sys.exit(0)
for nm, fn in VAR.items():
    CV = dict(cP)
    for c in KC: s_, g_ = fn(c); CV[c] = curve(c, s_, g_)
    tV = frozen_taus(CV); nV, _, _ = D.honest(CV, folds, frozen=tV)
    out = {"taus": [{c: {g: t[g][c] for g in sorted(t) if inl(g) and c in t[g]} for c in KC} for t in tV]}
    for sl, s in slices.items():
        a, b = M.score_counts(nB, s)[0], M.score_counts(nV, s)[0]; bs = M.paired_bootstrap_counts(nV, nB, site_family, BOOT, 0, s)
        out[sl] = {"B": a, "V": b, "delta": b - a, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    pa, pb = M.score_counts(nB)[1], M.score_counts(nV)[1]
    out["cls"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12} for c in pa if pa[c]["n_gt"] > 0}
    g = {"delta": out["all"]["delta"] >= 0.003, "P": (out["all"]["P"] or 0) >= PMIN, "new34": out["new34"]["delta"] >= -1e-12, "nolidar": out["nolidar"]["delta"] >= -1e-12,
         "damaged": out["cls"][PO]["d"] >= -1e-12, "intact": out["cls"][TS]["d"] >= -1e-12, "fort": out["cls"]["fortifikatsii"]["d"] >= -1e-12, "floors": all(v["ok"] for v in out["cls"].values())}
    g["PASS"] = all(g.values()); out["gate"] = g; res["variants"][nm] = out; save()
    log(f"{nm}F vs V3P: Δ {out['all']['delta']:+.6f} P {out['all']['P']} | new34 {out['new34']['delta']:+.6f} nolidar {out['nolidar']['delta']:+.6f} | "
        + " ".join(f"{c[:8]} {v['d']:+.5f}" for c, v in out["cls"].items()) + f" | {g}")
    del CV, nV; gc.collect()
log("done")
