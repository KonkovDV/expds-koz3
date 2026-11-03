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
OUT = Path("/workspace/koz3/orch/bot-W4/runs/v4on_archive_gate.json")
RPC = ("gorodishcha", "arkhitektury", "object_poly", "fortifikatsii")
res = {"prereg": "docs/PREREG_w4_v4_cloudfeat_nocnx.md (проверка через архив)", "npz": "nested_w5cfnc.npz", "boot": BOOT, "sanity": {}}
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

# --- archive check: blend from archive code (expds/w5_cloudfeat.py + ref.npz + v4_on config of the archive)
import tomllib
X = Path("/workspace/koz3/w4geo/x_v4on"); res["archive"] = "koz3_v4on_65149fe_aafe99d4_w644c0017.zip"
sp = importlib.util.spec_from_file_location("arch_w5cf", X / "expds/w5_cloudfeat.py"); AW = importlib.util.module_from_spec(sp); sp.loader.exec_module(AW)
cfg = tomllib.loads((X / "configs/default.toml").read_text(encoding="utf-8")); assert AW.enabled(cfg); res["alpha_archive"] = AW.spec(cfg)["alpha"]
def S_from(R):
    S = tconf.copy()
    for c, m in LIM.items():
        r = np.where(R[m] >= 0, R[m], np.nan); S[m] = AW.blend_scores(c, tconf[m], r, cfg)
    return S
SF = np.load("/workspace/koz3/orch/bot-W4/runs/w5cfnc_SF_train.npy")
SA = S_from(OUTS); d = np.abs(SA - SF); res["archive_blend_vs_fit_maxabs"] = float(d.max()); res["archive_blend_vs_fit_n_gt1e-9"] = int((d > 1e-9).sum())
log("archive blend vs fit", d.max(), (d > 1e-9).sum())
def cmp(nX, nR):
    o = {}
    for k, s in slices.items():
        a, b = M.score_counts(nR, s)[0], M.score_counts(nX, s)[0]
        bs = M.paired_bootstrap_counts(nX, nR, site_family, BOOT, 0, s)
        o[k] = {"ref": a, "var": b, "delta": b - a, "P": bs.get("p_gt0")}
    pa, pb = M.score_counts(nR)[1], M.score_counts(nX)[1]
    o["floor"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12}
                  for c in pa if pa[c]["n_gt"] > 0}
    return o
nA = D.honest(curves(SA), folds)[0]; res["arch_OOF_honest"] = cmp(nA, nB)
log("archive blend, OOF scores, honest-10", {k: (round(v["delta"], 6), v["P"]) for k, v in res["arch_OOF_honest"].items() if k != "floor"}); save()
# archive rankers (full-data, in-sample on train → optimistic, informative)
import lightgbm as lgb
meta = json.loads((X / "models/w5cf_infer/meta.json").read_text()); assert list(meta["CL"]) == CL, (meta["CL"], CL)
W5 = Path("/workspace/koz3/w1/cf/w5d/work"); X4 = np.load(W5 / "X4.npy", mmap_mode="r"); nm = json.load(open(W5 / "X4_names.json")); assert nm == meta["names"]
pool = OUTS >= 0; fam = np.where(tcls <= 1, 0, tcls); RI = np.full(N, -1.0)
for f_, v in meta["models"].items():
    ii = np.flatnonzero(pool & (fam == int(f_)))
    if not len(ii): continue
    Xi = np.array(X4[ii], np.float32); Xi[:, [nm.index(k) for k in nm if k.startswith("cnx_")]] = np.nan
    RI[ii] = lgb.Booster(model_file=str(X / "models/w5cf_infer" / v["file"])).predict(Xi, num_threads=4)
SI = S_from(RI); nI = D.honest(curves(SI), folds)[0]; res["arch_insample_honest"] = cmp(nI, nB)
from scipy.stats import spearmanr
res["spearman_insample_vs_oof_pool"] = float(spearmanr(RI[pool], OUTS[pool]).correlation)
log("archive rankers in-sample honest-10", {k: (round(v["delta"], 6), v["P"]) for k, v in res["arch_insample_honest"].items() if k != "floor"}, res["spearman_insample_vs_oof_pool"])
save(); log("done")
