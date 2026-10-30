#!/usr/bin/env python3
"""W4 P-D2-cleanroom (PREREG docs/PREREG_w4_d2_cleanroom.md): C=v2clean vs A; G4 C_iso vs C_merged."""
import importlib.util, json, sys, time
from collections import defaultdict, Counter
from pathlib import Path
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, P, log, site_family = G.D, G.M, G.P, G.log, G.site_family
from shapely.geometry import shape, mapping
OUT = Path("/workspace/koz3/orch/bot-W4/work/d2_gate.json"); BOOT = 2000
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10)
nA, _, _ = D.honest(cA, folds)
no_et = G.drop_engine(tab, lambda g, r: g == "etgr")
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
cC, _ = no_et.curves(c19); nC, _, _ = D.honest(cC, folds)
by = defaultdict(list)
for f in c19:
    q = f["properties"]; by[(q["region_name"], q["class_name"])].append(dict(q, geom=shape(f["geometry"])))
merged = []
for k in sorted(by):
    for m in P.merge_across_sources(by[k]):
        q = {kk: vv for kk, vv in m.items() if kk != "geom"}
        merged.append({"type": "Feature", "geometry": mapping(m["geom"]), "properties": q})
log(f"C etgr rows {len(c19)} -> merged across layers {len(merged)}")
cMg, _ = no_et.curves(merged); nMg, _, _ = D.honest(cMg, folds)
res = {"boot": BOOT, "n_etgr": {"iso": len(c19), "merged": len(merged)}}
def cmp(name, nB, nR, rname):
    sc = {}
    for k, s in slices.items():
        a_, b_ = M.score_counts(nR, s)[0], M.score_counts(nB, s)[0]
        bs = M.paired_bootstrap_counts(nB, nR, site_family, BOOT, 0, s)
        sc[k] = {rname: a_, name: b_, "delta": b_ - a_, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    pr, pb = M.score_counts(nR)[1], M.score_counts(nB)[1]
    pc = {c: {"f1_ref": pr[c]["f1"], "f1": pb[c]["f1"], "d": pb[c]["f1"] - pr[c]["f1"], "tp": [pr[c]["tp"], pb[c]["tp"]], "fp": [pr[c]["fp"], pb[c]["fp"]]} for c in pr if pr[c]["n_gt"] > 0}
    res[f"{name}_vs_{rname}"] = {"scores": sc, "per_class": pc}
    log(f"{name} vs {rname}: Δall {sc['all']['delta']:+.6f} P {sc['all']['P']} new34 {sc['new34']['delta']:+.6f} nolidar {sc['nolidar']['delta']:+.6f}; "
        + " ".join(f"{c} {v['d']:+.4f}" for c, v in pc.items()))
cmp("C", nC, nA, "A")
cmp("Ciso", nC, nMg, "Cmerged")
cmp("Cmerged", nMg, nA, "A")
d = res["C_vs_A"]["scores"]; pc = res["C_vs_A"]["per_class"]
res["D2_1_accept"] = {"delta_ge_0.009": d["all"]["delta"] >= 0.009, "P_ge_0.95": (d["all"]["P"] or 0) >= 0.95,
                      "slices_ge_-0.003": d["new34"]["delta"] >= -0.003 and d["nolidar"]["delta"] >= -0.003,
                      "finds_poly_not_below_A": all(pc[c]["d"] >= -1e-12 for c in ("finds_points", "object_poly") if c in pc)}
res["D2_1_accept"]["PASS"] = all(res["D2_1_accept"].values())
res["D2_2_parity_proxy"] = {"delta_C_A": d["all"]["delta"], "ref_old_line": 0.012, "abs_gap": abs(d["all"]["delta"] - 0.012),
                            "ok": abs(d["all"]["delta"] - 0.012) <= 0.002, "direct": "BLOCKED: v2 geophysics output (legacy PIXGBM) not available in clean room"}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str)); log(json.dumps({k: res[k] for k in ("D2_1_accept", "D2_2_parity_proxy")}))
