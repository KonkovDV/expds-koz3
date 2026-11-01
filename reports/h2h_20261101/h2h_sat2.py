#!/usr/bin/env python3
"""PREREG docs/PREREG_h2h_sat2_v3on.md (handoff 879453b): V3P vs S2 (B0 sat2 REPLACE) vs U (V3P+sat2), same engine."""
import importlib.util, json, sys, time, gc, copy
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, log, site_family = G.D, G.M, G.log, G.site_family
BOOT = 2000; SP = G.SP; KUR = {"kurgany_povrezhdennye", "kurgany_tselye"}
OUT = Path("/workspace/koz3/orch/bot-W4/runs/h2h_sat2_v3on.json")
def drop_c(tab, drop):
    t2 = copy.copy(tab); t2.base = {}
    for c, b in tab.base.items():
        keep = np.array([not drop(c, g, r) for g, r in zip(b["grp"], b["reg"])], bool)
        idx = np.flatnonzero(keep); new = {int(o): n for n, o in enumerate(idx)}
        t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx], pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    return t2
t0 = time.time()
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10); del cA
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
s2 = G.load_geo("/workspace/koz3/w1/sat2/pkg/data/sp_det2_oof_notta.geojson", lambda q: True)
assert all(f["properties"]["threshold_group"] == "sp_det2" and f["properties"]["class_name"] in KUR for f in s2)
s2 = [f for f in s2 if f["properties"]["region_name"] in regs]
dfx, dreg = G.dfine_feats()
res = {"prereg": "docs/PREREG_h2h_sat2_v3on.md@879453b", "boot": BOOT, "n_sat2": len(s2), "n_dfine": len(dfx)}
eB0 = drop_c(tab, lambda c, g, r: g == "etgr")
eV = drop_c(tab, lambda c, g, r: g == "etgr" or (g in SP and r in dreg))
eS = drop_c(tab, lambda c, g, r: g == "etgr" or (c in KUR and g in SP))
del tab; gc.collect()
H = lambda cv: D.honest(cv, folds)[0]
nB0 = H(eB0.curves(c19)[0]); nV = H(eV.curves(c19 + dfx)[0]); nS = H(eS.curves(c19 + s2)[0]); nU = H(eV.curves(c19 + dfx + s2)[0])
sc = lambda n, s=None: M.score_counts(n, s)[0]
res["scores"] = {k: {s: sc(n, sl) for s, sl in slices.items()} for k, n in [("B0", nB0), ("V3P", nV), ("S2", nS), ("U", nU)]}
log("scores", res["scores"]); assert abs(res["scores"]["V3P"]["all"] - 0.333787) < 5e-6, "V3P sanity"
def cmp(nX, nR):
    o = {}
    for k, s in slices.items():
        bs = M.paired_bootstrap_counts(nX, nR, site_family, BOOT, 0, s)
        o[k] = {"delta": sc(nX, s) - sc(nR, s), "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    pa, pb = M.score_counts(nR)[1], M.score_counts(nX)[1]
    o["floor"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12} for c in pa if pa[c]["n_gt"] > 0}
    return o
for name, a, b in [("S2_vs_B0", nS, nB0), ("V3P_vs_B0", nV, nB0), ("S2_vs_V3P", nS, nV), ("U_vs_V3P", nU, nV)]:
    res[name] = cmp(a, b); log(name, {k: (round(v["delta"], 6), v["P"]) for k, v in res[name].items() if k != "floor"}, all(v["ok"] for v in res[name]["floor"].values()))
g = res["U_vs_V3P"]
res["gate_U"] = {"delta": g["all"]["delta"] >= 0.003, "P95": (g["all"]["P"] or 0) >= 0.95, "floor": all(v["ok"] for v in g["floor"].values()),
                 "new34": g["new34"]["delta"] >= 0, "nolidar": g["nolidar"]["delta"] >= 0}
res["gate_U"]["PASS"] = all(res["gate_U"].values()); res["sec"] = round(time.time() - t0)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float)); log("gate_U", res["gate_U"], res["sec"], "s")
