#!/usr/bin/env python3
"""PREREG docs/PREREG_c1_gates_1101.md@1eb216d: G1 (sat2 kp-only) / G2 (li self_nms@8 non-kurgan) vs V3P in v3_on production context."""
import importlib.util, json, sys, time, gc, copy, pickle, tomllib
from collections import defaultdict
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, log, site_family = G.D, G.M, G.log, G.site_family
ARCH = "/tmp/km/v3arch"  # unzip of koz3_v3on_b505656_6d4764f8_w6dfd2e07.zip (expds/, configs/)
sys.path.insert(0, ARCH)
from expds import pipeline as PL
cfg = tomllib.load(open(ARCH + "/configs/default.toml", "rb"))
assert not cfg.get("w5_cloudfeat")
BOOT = 2000; SP = G.SP; KUR = {"kurgany_povrezhdennye", "kurgany_tselye"}; KP = "kurgany_povrezhdennye"; FORT = "fortifikatsii"
S2_FLOOR = PL.output_floors(cfg, "sp_or")[KP]; assert abs(S2_FLOOR - 0.3645) < 1e-9
NMS_R = 8.0
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/c1_gates/c1_gates.json")
FLc = {}
def fl(g, c):
    if g not in FLc: FLc[g] = PL.output_floors(cfg, g)
    return FLc[g].get(c, 0.15)
def sub(tab, masks):
    t2 = copy.copy(tab); t2.base = {}
    for c, b in tab.base.items():
        idx = np.flatnonzero(masks[c]); new = {int(o): n for n, o in enumerate(idx)}
        t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx], pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    return t2
t0 = time.time()
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
T = pickle.load(open("/workspace/koz3/w5data/work/table.pkl", "rb"))
Tcls = np.asarray(T["cls"]); Tmk = np.asarray(T["mk"]).astype(str); Tcx = np.asarray(T["cx"], float); Tcy = np.asarray(T["cy"], float)
# centroid sanity (cx/cy == UTM centroid)
GW = pickle.load(open("/workspace/koz3/w5data/work/geom_wkb.pkl", "rb")); import shapely.wkb as WKB
for i in np.linspace(0, len(Tcls) - 1, 300).astype(int):
    cc = WKB.loads(GW[i]).centroid; assert abs(cc.x - Tcx[i]) < 0.05 and abs(cc.y - Tcy[i]) < 0.05, i
del GW; gc.collect()
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10); del cA
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
s2 = G.load_geo("/workspace/koz3/w1/sat2/pkg/data/sp_det2_oof_notta.geojson", lambda q: True)
assert all(f["properties"]["threshold_group"] == "sp_det2" and f["properties"]["class_name"] in KUR for f in s2)
dfx, dreg = G.dfine_feats()
pf = lambda f, flo=None: f["properties"]["confidence"] >= (fl(f["properties"]["threshold_group"], f["properties"]["class_name"]) if flo is None else flo)
ex = [f for f in c19 + dfx if pf(f)]
s2kp = [f for f in s2 if f["properties"]["region_name"] in regs and f["properties"]["class_name"] == KP and pf(f, S2_FLOOR)]
res = {"prereg": "docs/PREREG_c1_gates_1101.md@1eb216d", "boot": BOOT, "arch": "koz3_v3on_b505656_6d4764f8_w6dfd2e07", "s2_floor": S2_FLOOR,
       "n_extras_full": len(c19) + len(dfx), "n_extras_pc": len(ex), "n_s2kp_pc": len(s2kp)}
# masks
mV, mR, mG2 = {}, {}, {}
nms_stat = defaultdict(lambda: [0, 0])
for k, c in enumerate(tab.CL):
    b = tab.base[c]; idx = np.flatnonzero(Tcls == k); assert len(idx) == len(b["conf"])
    d3 = np.array([g == "etgr" or (g in SP and r in dreg) for g, r in zip(b["grp"], b["reg"])], bool)
    flo = np.array([fl(g, c) for g in b["grp"]])
    mV[c] = ~d3; mR[c] = (~d3) & (b["conf"] >= flo); m2 = mR[c].copy()
    if c not in KUR:
        li = np.array([g.startswith("li") or mk.startswith("li") for g, mk in zip(b["grp"], Tmk[idx])], bool) & mR[c]
        byr = defaultdict(list)
        for n in np.flatnonzero(li): byr[b["reg"][n]].append(n)
        for r, ns in byr.items():
            ns = sorted(ns, key=lambda n: float(b["conf"][n]), reverse=True); taken = []
            for n in ns:
                x, y = Tcx[idx[n]], Tcy[idx[n]]
                if any((x - tx) ** 2 + (y - ty) ** 2 <= NMS_R * NMS_R for tx, ty in taken):
                    m2[n] = False; continue
                taken.append((x, y))
            nms_stat[c][0] += len(ns)
        nms_stat[c][1] = int((mR[c] & ~m2).sum())
    mG2[c] = m2
res["rows"] = {c: {"stack_v3p": int(mV[c].sum()), "stack_pc": int(mR[c].sum()), "g2_suppressed": int((mR[c] & ~mG2[c]).sum()),
                   "g2_li_pc": nms_stat[c][0] if c in nms_stat else 0} for c in tab.CL}
res["floors"] = FLc
log("rows", res["rows"])
eV, eR, eG2 = sub(tab, mV), sub(tab, mR), sub(tab, mG2)
del tab, T; gc.collect()
H = lambda cv: D.honest(cv, folds)[0]
nV = H(eV.curves(c19 + dfx)[0]); nR = H(eR.curves(ex)[0]); nG1 = H(eR.curves(ex + s2kp)[0]); nG2 = H(eG2.curves(ex)[0])
sc = lambda n, s=None: M.score_counts(n, s)[0]
res["scores"] = {k: {s: sc(n, sl) for s, sl in slices.items()} for k, n in [("V3P_full", nV), ("V3P_pc", nR), ("G1", nG1), ("G2", nG2)]}
log("scores", res["scores"]); assert abs(res["scores"]["V3P_full"]["all"] - 0.333787) < 5e-6, "V3P sanity"
def cmp(nX, nRf):
    o = {}
    for k, s in slices.items():
        bs = M.paired_bootstrap_counts(nX, nRf, site_family, BOOT, 0, s)
        o[k] = {"delta": sc(nX, s) - sc(nRf, s), "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    pa, pb = M.score_counts(nRf)[1], M.score_counts(nX)[1]
    o["cls"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "engine_floor_ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12}
                for c in pa if pa[c]["n_gt"] > 0}
    return o
res["V3Ppc_vs_V3Pfull"] = cmp(nR, nV)
for name, nX in [("G1", nG1), ("G2", nG2)]:
    o = cmp(nX, nR); res[name] = o
    gate = {"delta>=0.003": o["all"]["delta"] >= 0.003, "P>=0.975": (o["all"]["P"] or 0) >= 0.975,
            "new34>=-0.005": o["new34"]["delta"] >= -0.005, "damaged>=-0.005": o["cls"][KP]["d"] >= -0.005,
            "fort>=-0.005": o["cls"][FORT]["d"] >= -0.005}
    gate["PASS"] = all(gate.values()); gate["engine_class_floors_ok(report)"] = all(v["engine_floor_ok"] for v in o["cls"].values())
    res["gate_" + name] = gate
    log(name, round(o["all"]["delta"], 6), o["all"]["P"], gate)
res["sec"] = round(time.time() - t0)
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float)); log("done", res["sec"], "s")
