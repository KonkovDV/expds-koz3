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
OUT = Path("/workspace/koz3/orch/bot-W4/runs/independent_regate_W1J2_nocnx.json")
RPC = ("gorodishcha", "arkhitektury", "object_poly", "fortifikatsii")
res = {"prereg": "docs/PREREG_w4_v4_cloudfeat_nocnx.md", "npz": "nested_w5cfnc.npz", "boot": BOOT, "sanity": {}}
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
# --- 2. V4N nested estimate
J = {c: {} for c in CL}; picks = []
for k in range(K):
    t0 = time.time(); inner = [f for j, f in enumerate(folds) if j != k]; inn = {}
    for a in ALPHA:
        inn[a] = M.score_counts(D.honest(curves(blend(k, a)), inner)[0])[0]
    ak = max(ALPHA, key=lambda a: inn[a])
    n = D.honest(curves(blend(k, ak)), folds)[0]
    for c in n:
        for r, v in n[c].items():
            if r in folds[k]: J.setdefault(c, {})[r] = v
    picks.append({"alpha": ak, "inner": inn}); log("fold", k, picks[-1], f"{time.time()-t0:.0f}s"); res["picks_V4N"] = picks; save()
def cmp(nX, nR):
    o = {}
    for k, s in slices.items():
        a, b = M.score_counts(nR, s)[0], M.score_counts(nX, s)[0]
        bs = M.paired_bootstrap_counts(nX, nR, site_family, BOOT, 0, s)
        o[k] = {"ref": a, "var": b, "delta": b - a, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    pa, pb = M.score_counts(nR)[1], M.score_counts(nX)[1]
    o["floor"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12}
                  for c in pa if pa[c]["n_gt"] > 0}
    return o
g = cmp(J, nB)
gate = {"delta_ge_0.003": g["all"]["delta"] >= 0.003, "P_ge_0.9857": (g["all"]["P"] or 0) >= 0.9857,
        "floor_ok": all(v["ok"] for v in g["floor"].values()), "new34_ge0": g["new34"]["delta"] >= 0, "nolidar_ge0": g["nolidar"]["delta"] >= 0}
gate["PASS"] = all(gate.values())
res["V4N"] = M.score_counts(J)[0]; res["V4N_vs_V3P"] = g; res["gate_V4N"] = gate
log("V4N", res["V4N"], {k: (round(v["delta"], 6), v["P"]) for k, v in g.items() if k != "floor"}, gate)
save()
# --- 1. deploy: alpha = mode of J2 nested picks, full-data sweep (outer OOF everywhere)
pk = [p["alpha"] for p in picks]; cnt = Counter(pk); mx = max(cnt.values()); AD = min(a for a in cnt if cnt[a] == mx)
res["alpha_deploy"] = AD
def blend_full(a):
    S = tconf.copy(); ref = {}
    for c, m in LIM.items():
        S[m] = RB[m]; pm = m[OUTS[m] >= 0]
        ref[c] = dict(conf_li=np.sort(tconf[m]), rb_pool=np.sort(RB[pm]), r_pool=np.sort(OUTS[pm]))
        if len(pm) < 2: continue
        q = np.sort(RB[pm]); rt = q[(rankdata(OUTS[pm], method="ordinal") - 1).astype(int)]
        S[pm] = a * rt + (1 - a) * RB[pm]
    return S, ref
SF, REF = blend_full(AD)
np.save("/workspace/koz3/orch/bot-W4/runs/w5cfnc_SF_train.npy", SF)
sw = M.sweep_groups(curves(SF))
taus = {}
for c, v in sw.items():
    for g, gv in v["groups"].items(): taus.setdefault(g, {})[c] = float(gv["threshold"])
LIG = sorted(set(tgrp[li])); res["li_groups"] = LIG
res["thresholds_full"] = taus; res["thresholds_li_groups"] = {g: taus.get(g, {}) for g in LIG}
res["insample_f1_full"] = {c: float(v["f1"]) for c, v in sw.items()}
np.savez_compressed("/workspace/koz3/orch/bot-W4/runs/w5cfnc_ref.npz", alpha=np.float64(AD), classes=np.array(sorted(REF)),
                    **{f"{k}__{c}": v[k].astype(np.float64) for c, v in REF.items() for k in v})
log("alpha", AD, "li thresholds", res["thresholds_li_groups"]); save()
save(); log("done")
