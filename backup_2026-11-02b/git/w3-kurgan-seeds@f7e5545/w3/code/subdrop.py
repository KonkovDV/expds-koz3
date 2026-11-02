#!/usr/bin/env python3
"""KURG-SUBCLS gate (prereg/KURG_SUBCLS_PREREG.md). Clean-room. V3P context as kdual.py."""
import importlib.util, json, sys, gc, pickle
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, log, site_family = G.D, G.M, G.log, G.site_family
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = Path("/workspace/koz3/orch/bot-W3/runs/gate_SUBDROP.json")
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
T_reg, T_cls, T_CL = T["reg"], T["cls"], list(T["CL"]); G2T = {c: None for c in KC}
for k, c in enumerate(T["CL"]):
    if c in KC:
        idx = np.flatnonzero(T["cls"] == k); keep = np.array([not drop(g, r) for g, r in zip(tab.base[c]["grp"], tab.base[c]["reg"])], bool); G2T[c] = {int(t): p for p, t in enumerate(idx[keep])}
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
# ---------------- SUBCLS ----------------
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
PMIN = 0.9; MGRID = [None, 0.0, 0.05, 0.10, 0.20, 0.30]
z = np.load("/workspace/koz3/ds/kcrops/kurgan_subclass_crops.npz", allow_pickle=True)
cm = z["stack_idx"] >= 0
# pairs: key -> (pPO, pTS)
kp = {c: {k: p for p, k in enumerate(R[c]["key"]) if R[c]["li"][p]} for c in KC}
PAIR = {}; nmiss = 0
for ci in np.flatnonzero(cm):
    t = int(z["stack_idx"][ci]); c = str(z["cand_class"][ci])
    assert str(T_reg[t]) == str(z["region"][ci]) and T_CL[int(T_cls[t])] == c
    p = G2T[c].get(t)
    if p is None or not R[c]["li"][p]: nmiss += 1; continue
    k = R[c]["key"][p]; o = KC[1 - KC.index(c)]; q = kp[o].get(k)
    if q is None: nmiss += 1; continue
    pp = (p, q) if c == PO else (q, p)
    if pp not in PAIR: PAIR[pp] = ci
    elif z["y"][ci] >= 0 and z["y"][PAIR[pp]] < 0: PAIR[pp] = ci
res["sanity"]["crop_pairs"] = len(PAIR); res["sanity"]["crop_unmapped"] = nmiss
pairs = list(PAIR); crop = np.array([PAIR[p] for p in pairs]); NP = len(pairs)
preg = np.array([R[PO]["reg"][a] for a, _ in pairs]); y = z["y"][crop].astype(int)
# features
X3 = z["X"][crop].astype(np.float32); S_, L_, SL = (X3[:, 0] - 128) * 0.75 / 127, (X3[:, 1] - 128) * 2.0 / 127, X3[:, 2]
yy, xx = np.mgrid[0:64, 0:64]; dy, dx = yy - 31.5, xx - 31.5; rr = np.hypot(dx, dy) * 0.5; az = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
def m_(r0, r1): return (rr >= r0) & (rr <= r1)
def red(A, mask, f): return f(A[:, mask], axis=1)
F = {}
F["h"] = red(L_, m_(0, 4), np.max); F["h_s"] = red(S_, m_(0, 3), np.max)
F["vol"] = (np.clip(L_, 0, None)[:, m_(0, 8)]).sum(1) * 0.25
sec = np.stack([red(L_, m_(1, 8) & (az == a), np.mean) for a in range(8)], 1)
F["sym_var"] = sec.var(1); F["sym_rng"] = sec.max(1) - sec.min(1); F["sym_cv"] = sec.std(1) / (np.abs(sec.mean(1)) + 1e-3)
F["flat"] = red(L_, m_(0, 2), np.std) / (np.abs(F["h"]) + 1e-3); F["slope_top"] = red(SL, m_(0, 2), np.mean)
F["pit"] = red(S_, m_(0, 4), np.max) - red(S_, m_(0, 1), np.mean); F["pit_min"] = red(S_, m_(0, 2), np.min)
prof = np.stack([red(L_, m_(r, r + 0.5), np.mean) for r in np.arange(0, 12, 0.5)], 1)
F["edge"] = (-np.diff(prof, axis=1)).max(1); F["slope_ring"] = red(SL, m_(3, 8), np.mean); F["slope_ring90"] = np.percentile(SL[:, m_(3, 8)], 90, axis=1)
gy, gx = np.gradient(S_, axis=(1, 2))
def coh(mask):
    jxx, jyy, jxy = (gx * gx)[:, mask].sum(1), (gy * gy)[:, mask].sum(1), (gx * gy)[:, mask].sum(1)
    return np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)
F["aniso"] = coh(rr >= 0); F["aniso_ring"] = coh(m_(8, 16))
F["c_po"] = np.array([R[PO]["conf"][a] for a, _ in pairs]); F["c_ts"] = np.array([R[TS]["conf"][b] for _, b in pairs]); F["c_d"] = F["c_po"] - F["c_ts"]
GC = {g: i for i, g in enumerate(sorted(LIG))}; F["grp"] = np.array([GC[R[PO]["grp"][a]] for a, _ in pairs])
FN = list(F); XF = np.c_[[F[k] for k in FN]].T.astype(np.float64)
del X3, S_, L_, SL, gx, gy; gc.collect()
fold_of = {r: i for i, f in enumerate(folds) for r in f}; pf = np.array([fold_of.get(r, -1) for r in preg]); lab = y >= 0
PRM = dict(objective="binary", n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=4, verbose=-1)
def fitpred(tr, te):
    m = lgb.LGBMClassifier(**PRM).fit(XF[tr & lab], y[tr & lab]); return m.predict_proba(XF[te])[:, 1]
pO = np.full(NP, np.nan); PK = []
for k in range(len(folds)):
    held = pf == k; pk = np.full(NP, np.nan)
    if held.any(): pk[held] = pO[held] = fitpred(~held & (pf >= 0), held)
    for j in range(len(folds)):
        if j == k: continue
        hj = pf == j
        if hj.any(): pk[hj] = fitpred((pf >= 0) & (pf != k) & (pf != j), hj)
    PK.append(pk)
mm = lab & ~np.isnan(pO); auc = float(roc_auc_score(y[mm], pO[mm])); res["auc_oof"] = auc; res["auc_n"] = int(mm.sum())
log(f"crop pairs {NP} labelled {int(lab.sum())} OOF AUC {auc:.4f}"); save()
# coverage of V3P-emitted li rows
em = cov = 0; pset = {c: set(a if c == PO else b for a, b in pairs) for c in KC}
for k, held in enumerate(folds):
    for c in KC:
        A = R[c]; t = tB[k]
        for p in np.flatnonzero(A["li"][:A["nb"]]):
            if A["reg"][p] in held and A["conf"][p] >= t.get(A["grp"][p], {}).get(c, np.inf): em += 1; cov += p in pset[c]
res["sanity"]["emitted_li_rows"] = em; res["sanity"]["emitted_with_crop"] = cov; log(f"V3P-emitted li rows {em}, with crop {cov}"); save()
def curve2(c, conf, grp, keep):
    A = R[c]; idx = np.flatnonzero(keep); new = -np.ones(len(keep), np.int64); new[idx] = np.arange(len(idx))
    cf, rg, gp = conf[idx], A["reg"][idx], grp[idx]
    pr = {int(new[i]): v for i, v in A["pairs"].items() if keep[i]}
    order = np.argsort(-cf, kind="stable"); rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    hit = np.zeros(len(order), bool); used = set()
    for i in sorted(pr, key=lambda i: rank[i]):
        best, bj = float("-inf"), -1
        for j, s_ in sorted(pr[i]):
            if j not in used and s_ > best + 1e-9: best, bj = s_, j
        if bj >= 0: used.add(bj); hit[rank[i]] = True
    return M.ClassCurve(conf=cf[order], hit=hit, pred_region=rg[order].tolist(), n_gt=cP[c].n_gt, gt_regions=cP[c].gt_regions, pred_group=gp[order].tolist())
def variant(p, m):   # SUBDROP: keep conf/group, drop the disagreeing twin when |p-0.5| >= m
    out = {}; dmg = p >= 0.5; sure = (~np.isnan(p)) & (np.abs(p - 0.5) >= m)
    for c in KC:
        keep = np.ones(len(R[c]["conf"]), bool); pos = np.array([a if c == PO else b for a, b in pairs])
        keep[pos[sure & ((~dmg) if c == PO else dmg)]] = False
        out[c] = curve2(c, R[c]["conf"], R[c]["grp"], keep)
    return out
def cnt(curves, taus, regset):
    out = {}
    for c in [c for c in KC if c in curves]:
        cv = curves[c]; per = {r: [0, 0, int(k)] for r, k in cv.gt_regions.items() if r in regset}
        for x, h, r, g in zip(cv.conf, cv.hit, cv.pred_region, cv.pred_group):
            if r in regset and x >= taus.get(g, {}).get(c, np.inf):
                l = per.setdefault(r, [0, 0, 0]); l[0 if h else 1] += 1
        out[c] = per
    return out
every = set().union(*folds)
for c in KC:   # sanity: cnt reproduces nB
    t0 = cnt({c: cP[c]}, tB[0], folds[0])[c]
    assert all(t0.get(r, [0, 0, 0]) == nB[c].get(r, [0, 0, 0]) for r in set(t0) | {r for r in nB[c] if r in folds[0]}), c
nV = {c: {r: list(v) for r, v in by.items()} for c, by in nB.items()}; chosen = []; trs = []
for k, held in enumerate(folds):
    train = every - held; best, bs, bh = None, M.score_counts(cnt(cP, tB[k], train))[0], None
    row = {"off": bs}
    for m in MGRID[1:]:
        CV = variant(PK[k], m); _, _, tk = D.honest(CV, [held, train]); s = M.score_counts(cnt(CV, tk[0], train))[0]; row[str(m)] = s
        if s > bs + 1e-12 or (best is not None and abs(s - bs) <= 1e-12): best, bs, bh = m, s, cnt(CV, tk[0], held)
        del CV; gc.collect()
    chosen.append(best); trs.append(row); log(f"fold {k} m={best} train {row}")
    if bh is not None:
        for c in KC:
            for r in list(nV[c]):
                if r in held: del nV[c][r]
            nV[c].update(bh[c])
res["m_per_fold"] = chosen; res["train_scores"] = trs; save()
out = {}
for sl, s in slices.items():
    a, b = M.score_counts(nB, s)[0], M.score_counts(nV, s)[0]; bs = M.paired_bootstrap_counts(nV, nB, site_family, BOOT, 0, s)
    out[sl] = {"B": a, "V": b, "delta": b - a, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
pa, pb = M.score_counts(nB)[1], M.score_counts(nV)[1]
out["cls"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12} for c in pa if pa[c]["n_gt"] > 0}
g = {"delta": out["all"]["delta"] >= 0.003, "P": (out["all"]["P"] or 0) >= PMIN, "new34": out["new34"]["delta"] >= -1e-12, "nolidar": out["nolidar"]["delta"] >= -1e-12,
     "damaged": out["cls"][PO]["d"] >= -1e-12, "intact": out["cls"][TS]["d"] >= -0.002 - 1e-12, "fort": out["cls"]["fortifikatsii"]["d"] >= -1e-12, "floors": all(v["ok"] for v in out["cls"].values())}
g["PASS"] = all(g.values()); out["gate"] = g; res["gate"] = out; save()
log(f"SUBDROP vs V3P: AUC {auc:.4f} Δ {out['all']['delta']:+.6f} P {out['all']['P']} | new34 {out['new34']['delta']:+.6f} nolidar {out['nolidar']['delta']:+.6f} | "
    + " ".join(f"{c[:8]} {v['d']:+.5f}" for c, v in out["cls"].items()) + f" | m {chosen} | {g}")
