#!/usr/bin/env python3
"""PREREG docs/PREREG_w4_v4_cloudfeat_nocnx.md: V4N gate (nocnx) + deploy alpha/thresholds."""
import importlib.util, json, sys, time, gc, math, copy, pickle
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, log, site_family = G.D, G.M, G.log, G.site_family
import shapely.geometry as sg
BOOT = 2000; ALPHA = [0.3, 0.6, 1.0]
OUT = Path("/workspace/koz3/orch/bot-W1/work/j2nc/ctxgate.json")
RPC = ("gorodishcha", "arkhitektury", "object_poly", "fortifikatsii")
res = {"what": "W1 production-context gate (J2 nocnx)", "npz": "nested_w5cfnc.npz", "boot": BOOT, "sanity": {}}
def save(): OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
T = pickle.load(open("/workspace/koz3/w5data/work/table.pkl", "rb"))
CL = list(T["CL"]); cid = {c: i for i, c in enumerate(CL)}
tcls = T["cls"].astype(int); tconf = T["conf"].astype(np.float64); treg = T["reg"].astype(str); tgrp = T["grp"].astype(str); tmk = T["mk"].astype(str)
N = len(tcls)
greg_regs = {r for c in CL for r in T["greg"][c]}
del T["pairs"]; gc.collect()
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10)
regs_ws = set(treg) | greg_regs
res["sanity"]["folds_equal_ws_rr"] = M.family_folds(regs_ws, site_family, 10) == folds
assert res["sanity"]["folds_equal_ws_rr"]
fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in treg]); K = len(folds)
# nested OOF (W1 data) + blend exactly as ws_rr
z = np.load("/workspace/koz3/w1/cf/w5d/work/nested_w5cfnc.npz"); OUTS = z["outer"]; INN = z["inner"]
li = tmk == "li"; RB = np.zeros(N); LIM = {}
for c in CL:
    m = np.flatnonzero(li & (tcls == cid[c]))
    if len(m): RB[m] = (rankdata(tconf[m]) - 0.5) / len(m); LIM[c] = m
def blend(k, a):
    R = np.where(fid == k, OUTS, INN[k]); S = tconf.copy()
    for c, m in LIM.items():
        S[m] = RB[m]; pm = m[R[m] >= 0]
        if len(pm) < 2: continue
        q = np.sort(RB[pm]); rt = q[(rankdata(R[pm], method="ordinal") - 1).astype(int)]
        S[pm] = a * rt + (1 - a) * RB[pm]
    return S
# V3P (B)
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg = G.dfine_feats(); SP = G.SP
drop3 = lambda g, r: g == "etgr" or (g in SP and r in dreg)
base3 = G.drop_engine(tab, drop3)
cP, _ = base3.curves(c19 + dfx); nB = D.honest(cP, folds)[0]; del cP
res["V3P"] = M.score_counts(nB)[0]; log("V3P", res["V3P"])
# stack RPC rows with table index
rows = []
for p in sorted(Path("/workspace/koz3/w1/d3/split_arch").glob("*.tsv")):
    for line in open(p, encoding="utf-8"):
        if not any(f'"{c}"' in line for c in RPC): continue
        key, _, js = line.partition("\t"); f = json.loads(js)
        if f["properties"].get("class_name") in RPC: f["_tid"] = int(key); rows.append(f)
kbad = sum(1 for f in rows if CL[tcls[f["_tid"]]] != f["properties"]["class_name"] or treg[f["_tid"]] != f["properties"]["region_name"]
           or abs(tconf[f["_tid"]] - float(f["properties"]["confidence"])) > 1e-6)
res["sanity"]["K_key_mismatch"] = kbad; log("K mismatch", kbad); assert kbad == 0
grp = lambda f: M.threshold_group(f["properties"])
rows3 = [f for f in rows if not drop3(grp(f), f["properties"]["region_name"])]
extras = rows3 + c19 + dfx  # no M_rp
etid = [f.get("_tid", -1) for f in rows3] + [-1] * (len(c19) + len(dfx))
ex = base3._extra(extras)
# alignment of extras with table index, per class (input order)
ecl = [f["properties"]["class_name"] for f in extras]; EXT = {}
for c in CL:
    ii = [i for i, cc in enumerate(ecl) if cc == c]
    e = ex.get(c)
    if e is None: assert not ii, c; continue
    assert len(ii) == len(e["conf"]), (c, len(ii), len(e["conf"]))
    t = np.array([etid[i] for i in ii]); exp = np.array([float(extras[i]["properties"]["confidence"]) for i in ii])
    assert np.allclose(e["conf"], exp, atol=1e-9), c
    EXT[c] = t
# base (non-RPC) table index
BT = {}
for c in CL:
    if c in RPC: continue
    idx = np.flatnonzero(tcls == cid[c]); keep = np.array([not drop3(g, r) for g, r in zip(tgrp[idx], treg[idx])], bool)
    BT[c] = idx[keep]; assert np.array_equal(base3.base[c]["conf"], tconf[BT[c]])
def curves(S):
    out = {}
    for c in CL:
        if c in RPC:
            b = dict(conf=np.zeros(0), reg=np.zeros(0, str), grp=np.zeros(0, str), pairs={}); bt = np.zeros(0, int)
        else:
            b = base3.base[c]; bt = BT[c]
        e = ex.get(c); nb = len(b["conf"])
        cb = np.where(li[bt], S[bt], b["conf"]) if nb else b["conf"]
        if e is not None:
            t = EXT[c]; ce = e["conf"].copy(); mm = t >= 0
            ce[mm] = np.where(li[t[mm]], S[t[mm]], ce[mm])
            conf = np.concatenate([cb, ce]); reg = np.concatenate([b["reg"], e["reg"]]); gp = np.concatenate([b["grp"], e["grp"]])
            pairs = dict(b["pairs"]); pairs.update({nb + i: v for i, v in e["pairs"].items()})
        else:
            conf, reg, gp, pairs = cb, b["reg"], b["grp"], dict(b["pairs"])
        order = np.argsort(-conf, kind="stable"); rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
        hit = np.zeros(len(order), bool); used = set()
        for i in sorted(pairs, key=lambda i: rank[i]):
            best, bj = float("-inf"), -1
            for j, s_ in sorted(pairs[i]):
                if j not in used and s_ > best + 1e-9: best, bj = s_, j
            if bj >= 0: used.add(bj); hit[rank[i]] = True
        out[c] = M.ClassCurve(conf=conf[order], hit=hit, pred_region=reg[order].tolist(), n_gt=len(tab.G[c]),
                              gt_regions=Counter(tab.greg[c]), pred_group=gp[order].tolist())
    return out
nP = D.honest(curves(tconf), folds)[0]; sP = M.score_counts(nP)[0]
res["sanity"]["V3P_via_curves"] = sP; res["sanity"]["V3P_minus_engine"] = sP - res["V3P"]; log("V3P via curves", sP, sP - res["V3P"]); save()
assert abs(sP - res["V3P"]) <= 1e-9

from scipy.stats import ks_2samp
def cmp(nX, nR):
    o = {}
    for k, s in slices.items():
        a, b = M.score_counts(nR, s)[0], M.score_counts(nX, s)[0]
        bs = M.paired_bootstrap_counts(nX, nR, site_family, BOOT, 0, s)
        o[k] = {"delta": b - a, "P": bs.get("p_gt0")}
    pa, pb = M.score_counts(nR)[1], M.score_counts(nX)[1]
    o["floor"] = {c: round(pb[c]["f1"] - pa[c]["f1"], 6) for c in pa if pa[c]["n_gt"] > 0}
    return o
def short(o): return {k: (round(v["delta"], 6), v["P"]) for k, v in o.items() if k != "floor"} | {"floor": o["floor"]}
def mixR(k, cls_mixed=None):
    """R for evaluating fold k: eval rows OUTS; training rows INN[k] (nested) for classes in cls_mixed, OUTS otherwise."""
    R = OUTS.copy()
    tr = fid != k
    if cls_mixed is None: R[tr] = INN[k][tr]
    else:
        for c in cls_mixed:
            mm = tr & (tcls == cid[c]); R[mm] = INN[k][mm]
    return R
def S_of(R, a):
    S = tconf.copy()
    for c, m in LIM.items():
        S[m] = RB[m]; pm = m[R[m] >= 0]
        if len(pm) < 2: continue
        q = np.sort(RB[pm]); rt = q[(rankdata(R[pm], method="ordinal") - 1).astype(int)]
        S[pm] = (a[c] if isinstance(a, dict) else a) * rt + (1 - (a[c] if isinstance(a, dict) else a)) * RB[pm]
    return S
V = {}
def run(name, J):
    V[name] = J; res.setdefault('variants', {})[name] = cmp(J, nB); log(name, short(res['variants'][name])); save()

# ---------------- production-like context (exB.py) ----------------
import lightgbm as lgb, os
J = "/workspace/koz3/orch/bot-W1/work/j2nc/"; W5 = "/workspace/koz3/w1/cf/w5d/work/"
cm = np.load(J + "ctx_masks.npz"); keep = cm["keep"]; srow = cm["score_rows"]; Xp = np.load(J + "Xp.npy")
X = np.load(W5 + "X4nc.npy"); names = json.load(open(W5 + "X4nc_names.json"))
pool = np.load(W5 + "pool.npy"); y = np.load(W5 + "first.npy")
fam = np.where(tcls <= 1, 0, tcls)
mono = [1 if n in ("conf", "logit") else 0 for n in names]; cat = [names.index(n) for n in ("cls", "grp", "mk") if n in names]
P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=1.0, monotone_constraints=mono, verbose=-1, num_threads=4, seed=0)
cache = Path(J + "ctx_scores.npz")
if cache.exists():
    z = np.load(cache); O_re, O_prod, O_v = z["O_re"], z["O_prod"], z["O_v"]
else:
    O_re = np.full(N, -1.0); O_prod = np.full(N, np.nan); O_v = np.full(N, np.nan)
    for k in range(K):
        t = time.time()
        for f in np.unique(fam[tcls >= 0]):
            tr = (fam == f) & (fid != k) & pool
            pm = (fam == f) & (fid == k) & pool; ps = (fam == f) & (fid == k) & srow
            if y[tr].sum() < 5:
                O_re[pm] = tconf[pm]; O_prod[ps] = tconf[ps]
            else:
                m = lgb.train(P, lgb.Dataset(X[tr], y[tr].astype(float), categorical_feature=cat), 300)
                O_re[pm] = m.predict(X[pm]); 
                if ps.any(): O_prod[ps] = m.predict(Xp[ps])
            trv = (fam == f) & (fid != k) & srow
            if ps.any():
                if y[trv].sum() < 5: O_v[ps] = tconf[ps]
                else:
                    mv = lgb.train(P, lgb.Dataset(Xp[trv], y[trv].astype(float), categorical_feature=cat), 300)
                    O_v[ps] = mv.predict(Xp[ps])
        log(f"fold {k} {time.time()-t:.0f}s")
    np.savez(cache, O_re=O_re, O_prod=O_prod, O_v=O_v)
pm_all = pool & (OUTS >= 0)
res["sanity"]["retrain_vs_OUTS_maxabs"] = float(np.abs(O_re[pm_all] - OUTS[pm_all]).max())
from scipy.stats import spearmanr
res["sanity"]["retrain_vs_OUTS_spearman"] = float(spearmanr(O_re[pm_all], OUTS[pm_all])[0])
log("retrain sanity", res["sanity"]["retrain_vs_OUTS_maxabs"], res["sanity"]["retrain_vs_OUTS_spearman"])
sh = {}
for c, m in LIM.items():
    mm = m[srow[m]]
    if len(mm) > 5:
        sh[c] = dict(n_surv=int(len(mm)), n_li=int(len(m)), spearman_full_vs_prod=float(spearmanr(OUTS[mm], O_prod[mm])[0]),
                     mean_shift=float((O_prod[mm] - OUTS[mm]).mean()), mad=float(np.abs(O_prod[mm] - OUTS[mm]).mean()))
res["oof_shift_on_survivors"] = sh; log("shift", sh)
def S_ref(Rsurv, ref_rows_fn, a=0.6, drop=True, Rref=None):
    """blend_scores-equivalent: survivors with score -> a*q[idx]+(1-a)*RB; non-survivor li rows dropped (production early floor)."""
    S = tconf.copy()
    for c, m in LIM.items():
        S[m] = RB[m]
        rr = ref_rows_fn(m)
        q = np.sort(RB[rr]); rp = np.sort(Rref[rr])
        sm = m[srow[m] & np.isfinite(Rsurv[m]) & (Rsurv[m] >= 0)]
        if len(rp) >= 2 and len(sm):
            idx = np.clip(np.searchsorted(rp, Rsurv[sm], side="right") - 1, 0, len(q) - 1)
            S[sm] = a * q[idx] + (1 - a) * RB[sm]
        if drop: S[m[~keep[m]]] = -1.0
    return S
full_pool = lambda m: m[OUTS[m] >= 0]
run("G0_archive_D3", D.honest(curves(S_of(OUTS, 0.6)), folds)[0])
run("G0b_full_ctx_drop_nonsurv", D.honest(curves(S_ref(OUTS, full_pool, Rref=OUTS)), folds)[0])
run("G1_prod_ctx_current_v4", D.honest(curves(S_ref(O_prod, full_pool, Rref=OUTS)), folds)[0])
log("done C")
