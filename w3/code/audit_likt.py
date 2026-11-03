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
OUT = Path("/workspace/koz3/orch/bot-W3/runs/audit_LIKT.json")
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

inl = lambda g: g.split("|")[0] in LIG
au = {}
def split_counts(CV, tk_list, regset):
    out = {}
    for c in KC:
        cv = CV[c]; d = {"li": [0, 0], "other": [0, 0]}
        for k, held in enumerate(folds):
            hs = held & regset
            for x, h, r, g in zip(cv.conf, cv.hit, cv.pred_region, cv.pred_group):
                if r in hs and x >= tk_list[k].get(g, {}).get(c, np.inf): d["li" if inl(g) else "other"][0 if h else 1] += 1
        out[c] = d
    return out
new = set(stk.new)
base_split = split_counts(cP, tB, new)
for nm in ("A", "C"):
    CV = dict(cP)
    for c in KC: s_, g_ = VAR[nm](c); CV[c] = curve(c, s_, g_)
    nV, _, tV = D.honest(CV, folds)
    ch = []
    for k in range(len(folds)):
        for c in KC:
            for g in sorted(set(tB[k]) | set(tV[k])):
                if inl(g): continue
                a, b = tB[k].get(g, {}).get(c), tV[k].get(g, {}).get(c)
                if a != b: ch.append([k, c[:8], g, a, b])
    perfold = []
    for k, held in enumerate(folds):
        hs = held & new
        perfold.append(M.score_counts({c: {r: v for r, v in nV[c].items() if r in hs} for c in nV})[0] - M.score_counts({c: {r: v for r, v in nB[c].items() if r in hs} for c in nB})[0] if hs else None)
    au[nm] = {"out_of_layer_tau_changes": ch, "n_changes": len(ch), "folds_changed": sorted({x[0] for x in ch}),
              "new34_tp_fp_base": base_split, "new34_tp_fp_var": split_counts(CV, tV, new), "new34_delta_per_fold_restricted": perfold,
              "new34_delta": M.score_counts(nV, stk.new)[0] - M.score_counts(nB, stk.new)[0]}
    log(f"AUDIT {nm}: {len(ch)} out-of-layer tau changes in folds {au[nm]['folds_changed']}; new34 split base {base_split} var {au[nm]['new34_tp_fp_var']}")
    res["audit"] = au; save(); del CV, nV; gc.collect()
log("audit done")
