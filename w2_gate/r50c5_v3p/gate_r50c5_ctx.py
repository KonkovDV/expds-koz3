#!/usr/bin/env python3
"""P-W2-9 amendment 2: li_r50c5 on V3P in v3_on PROD CONTEXT (output_floors of the v3_on archive config).
Engine = W4 v3_gate_lib (d3_gate TableEngine, honest-10 family folds, paired bootstrap 2000 by site_family, seed 0).
Run: nice -n 10 /workspace/koz3/w4venv/bin/python gate_r50c5_ctx.py [BOOT] [OUT]"""
import importlib.util, json, sys, time, gc, tomllib
from collections import Counter
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
sys_argv = sys.argv; sys.argv = sys.argv[:1]
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G); sys.argv = sys_argv
D, M, log, site_family = G.D, G.M, G.log, G.site_family
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "gate_r50c5_ctx.json"
V3P_REF = 0.3337867244
CFG_P = HERE / "v3on_src/configs/default.toml"   # extracted from w4geo/dist/koz3_v3on_b505656_6d4764f8_w6dfd2e07.zip
R = Path("/workspace/koz3/w2/oof/r50c5")
OOF = [R / "elenakonkova_koz3-w2-cnx-r50c5-f01/oof_cnx_c5-sc_f0.geojson", R / "elenakonkova_koz3-w2-cnx-r50c5-f01/oof_cnx_c5-sc_f1.geojson",
       R / "dmitrijkv_koz3-w2-cnx-r50c5-f234/oof_cnx_c5-sc_f2.geojson", R / "dmitrijkv_koz3-w2-cnx-r50c5-f234/oof_cnx_c5-sc_f3.geojson",
       R / "dmitrijkv_koz3-w2-cnx-r50c5-f234/oof_cnx_c5-sc_f4.geojson"]
cfg = tomllib.load(open(CFG_P, "rb"))
post = cfg.get("post") or {}
assert not post.get("kurgan_neighbour_prior") and not post.get("site_calibration") and "w5_cloudfeat" not in cfg
_FL = {}
def floor(g, c):   # == expds/pipeline.output_floors(cfg, g)[c] of v3_on (no prior/calib active)
    if g not in _FL:
        thr, own, dft = cfg.get("thresholds", {}), cfg.get("thresholds_by_markup", {}).get(g, {}), cfg.get("default_threshold", 0.15)
        _FL[g] = {cc: float(own.get(cc, thr.get(cc, dft))) for cc in D.CLASSES5 + ("object_poly", "finds_points")}
    return _FL[g].get(c, float(cfg.get("default_threshold", 0.15)))

def ctx_engine(tab, extra_floor=None):
    import copy
    t2 = copy.copy(tab); t2.base = {}; n0 = n1 = 0
    for c, b in tab.base.items():
        keep = np.array([cf >= floor(g, c) for g, cf in zip(b["grp"], b["conf"])], bool)
        n0 += len(keep); n1 += int(keep.sum())
        idx = np.flatnonzero(keep); new = {int(o): n for n, o in enumerate(idx)}
        t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx], pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    return t2, n0, n1

def fl_feats(fs, override=None):
    out = []
    for f in fs:
        q = f["properties"]; g = M.threshold_group(q)
        t = override if (override is not None) else floor(g, q["class_name"])
        if float(q["confidence"]) >= t: out.append(f)
    return out

def main():
    t0 = time.perf_counter(); res = {"what": "P-W2-9 amendment 2: li_r50c5 on V3P, v3_on prod context", "boot": BOOT, "cfg": str(CFG_P), "variants": {}}
    save = lambda: OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
                  Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
    tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
    slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
    cA, _ = tab.curves()
    regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
    folds = M.family_folds(regs, site_family, 10); del cA
    c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
    dfx, dreg = G.dfine_feats(); SP = G.SP
    base3 = G.drop_engine(tab, lambda g, r: g == "etgr" or (g in SP and r in dreg)); del tab; gc.collect()
    # S0: V3P exact
    cP, _ = base3.curves(c19 + dfx); nP = D.honest(cP, folds)[0]; del cP
    res["V3P"] = M.score_counts(nP)[0]; res["S0_V3P_minus_ref"] = res["V3P"] - V3P_REF; res["S0_ok"] = abs(res["V3P"] - V3P_REF) <= 1e-6
    log(f"S0 V3P {res['V3P']:.10f} d {res['S0_V3P_minus_ref']:+.2e} ok {res['S0_ok']}"); save()
    assert res["S0_ok"], "V3P not reproduced"
    # r50c5 rows
    r50 = []
    for p in OOF:
        for f, _ in D.iter_features(p):
            q = f["properties"]
            if q.get("class_name") not in D.CLASSES5: continue
            q["threshold_group"] = "li_r50c5"; q["markup_type"] = "li"; r50.append(f)
    res["n_r50c5_all"] = len(r50); res["r50c5_classes_all"] = dict(Counter(f["properties"]["class_name"] for f in r50))
    # A = V3P_ctx
    bctx, n0, n1 = ctx_engine(base3)
    ex_ctx = fl_feats(c19 + dfx)
    res["ctx_rows"] = {"stack_before": n0, "stack_kept": n1, "extras_before": len(c19) + len(dfx), "extras_kept": len(ex_ctx)}
    res["floors_used"] = _FL
    cA, _ = bctx.curves(ex_ctx); nA = D.honest(cA, folds)[0]
    gA = sorted({g for c in cA.values() for g in c.pred_group}); del cA
    res["V3P_ctx"] = M.score_counts(nA)[0]; log(f"V3P_ctx {res['V3P_ctx']:.10f} rows {res['ctx_rows']}"); save()

    def score(name, nB, nRef, gate=False):
        sc = {}
        for k, s in slices.items():
            sa, sb = M.score_counts(nRef, s)[0], M.score_counts(nB, s)[0]
            bs = M.paired_bootstrap_counts(nB, nRef, site_family, BOOT, 0, s) if BOOT else {}
            sc[k] = {"A": sa, "B": sb, "delta": sb - sa, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
        pa, pb = M.score_counts(nRef)[1], M.score_counts(nB)[1]
        fl = {c: {"f1_A": pa[c]["f1"], "f1_B": pb[c]["f1"], "d": pb[c]["f1"] - pa[c]["f1"], "n_gt": pa[c]["n_gt"],
                  "ok": (pb[c]["f1"] - pa[c]["f1"] >= -0.005 - 1e-12) or pa[c]["n_gt"] < 5,
                  "tp": [pa[c]["tp"], pb[c]["tp"]], "fp": [pa[c]["fp"], pb[c]["fp"]]} for c in pa if pa[c]["n_gt"] > 0}
        g = {"delta_ge_0.003": sc["all"]["delta"] >= 0.003, "P_ge_0.95": (sc["all"]["P"] or 0) >= 0.95,
             "class_floors_ge_-0.005": all(v["ok"] for v in fl.values()),
             "new34_ge_-0.001": sc["new34"]["delta"] >= -0.001, "nolidar_ge_-0.001": sc["nolidar"]["delta"] >= -0.001}
        g["PASS"] = all(g.values())
        res["variants"][name] = {"decisive": gate, "scores": sc, "floor": fl, "gate": g}
        log(f"{name}: A {sc['all']['A']:.6f} B {sc['all']['B']:.6f} Δ {sc['all']['delta']:+.6f} P {sc['all']['P']} | new34 {sc['new34']['delta']:+.6f} "
            f"| nolidar {sc['nolidar']['delta']:+.6f} | floors {{{', '.join(f'{c[:6]} {v['d']:+.4f}' for c, v in fl.items())}}} | PASS {g['PASS']}"); save()

    # B (decisive)
    rB = fl_feats(r50); res["n_r50c5_ctx"] = len(rB); res["r50c5_classes_ctx"] = dict(Counter(f["properties"]["class_name"] for f in rB))
    cB, _ = bctx.curves(ex_ctx + rB); nB = D.honest(cB, folds)[0]; del cB
    score("B_ctx_vs_V3P_ctx", nB, nA, gate=True)
    score("B_ctx_vs_V3P", nB, nP)
    # report-only: r50c5 floor 0.05
    r05 = fl_feats(r50, 0.05); cB5, _ = bctx.curves(ex_ctx + r05); nB5 = D.honest(cB5, folds)[0]; del cB5
    score("B_ctx_r50floor005_vs_V3P_ctx", nB5, nA)
    # report-only: fixed archive thresholds (all groups held at their floors, no sweep)
    cAf, _ = bctx.curves(ex_ctx); fx = set(gA) | {"li_r50c5"}
    nAf = D.honest(cAf, folds, fixed=fx)[0]; del cAf
    cBf, _ = bctx.curves(ex_ctx + rB); nBf = D.honest(cBf, folds, fixed=fx)[0]; del cBf
    res["fixed_A"] = M.score_counts(nAf)[0]
    score("fixed_archive_thr_B_vs_A", nBf, nAf)
    res["sec"] = round(time.perf_counter() - t0); save(); log("done", res["sec"], "s")

if __name__ == "__main__":
    main()
