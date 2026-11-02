#!/usr/bin/env python3
"""P-W2-12: soft map blend li_cnxblend (a) blend3 c3-sc s012, (b) blend4 s012+r50c5 on V3P in v3_on b447649 PROD CONTEXT.
Copied from w2_gate/r50c5_v3p/gate_r50c5_ctx.py (P-W2-9 amendment 2); only the extras layer, PASS rules and outputs differ.
Engine = W4 v3_gate_lib (d3_gate TableEngine, honest-10 family folds, paired bootstrap 2000 by site_family, seed 0).
Run: nice -n 10 /workspace/koz3/w4venv/bin/python gate_blend_ctx.py [BOOT] [OUT]"""
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
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "gate_blend_ctx.json"
V3P_REF = 0.3337867244
CFG_P = HERE / "v3on_src/configs/default.toml"   # extracted from w4geo/dist/koz3_v3on_b447649_6d4764f8_w6dfd2e07.zip (== b505656 byte-for-byte)
B = Path("/workspace/koz3/w2/blend")
S2 = Path("/workspace/koz3/w2/oof/s2")
VAR = {"a_blend3": [B / "blend3.geojson"], "b_blend4": [B / "blend4.geojson"],
       "ref_s2_single": [S2 / f"koz3-w2-cnx-c3sc-s2f{x}/oof_cnx_c3-sc_f{k}.geojson" for x, ks in (("01", (0, 1)), ("23", (2, 3)), ("4", (4,))) for k in ks]}
DECISIVE = ("a_blend3", "b_blend4")
P_MIN = 0.95          # standard 0.9, Bonferroni x2 -> alpha 0.10/2
FLOOR = {"kurgany_povrezhdennye": -0.002, "kurgany_tselye": -0.002, "gorodishcha": -0.002}   # others -0.005
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
    t0 = time.perf_counter(); res = {"what": "P-W2-12: soft map blend on V3P, v3_on b447649 prod context", "boot": BOOT, "cfg": str(CFG_P),
                                     "P_min": P_MIN, "variants": {}}
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
    cP, _ = base3.curves(c19 + dfx); nP = D.honest(cP, folds)[0]; del cP
    res["V3P"] = M.score_counts(nP)[0]; res["S0_V3P_minus_ref"] = res["V3P"] - V3P_REF; res["S0_ok"] = abs(res["V3P"] - V3P_REF) <= 1e-6
    log(f"S0 V3P {res['V3P']:.10f} d {res['S0_V3P_minus_ref']:+.2e} ok {res['S0_ok']}"); save()
    assert res["S0_ok"], "V3P not reproduced"
    bctx, n0, n1 = ctx_engine(base3)
    ex_ctx = fl_feats(c19 + dfx)
    res["ctx_rows"] = {"stack_before": n0, "stack_kept": n1, "extras_before": len(c19) + len(dfx), "extras_kept": len(ex_ctx)}
    res["floors_used"] = _FL
    cA, _ = bctx.curves(ex_ctx); nA = D.honest(cA, folds)[0]; del cA
    res["V3P_ctx"] = M.score_counts(nA)[0]; log(f"V3P_ctx {res['V3P_ctx']:.10f} rows {res['ctx_rows']}"); save()

    def score(name, nB, nRef, gate=False, n_rows=None):
        sc = {}
        for k, s in slices.items():
            sa, sb = M.score_counts(nRef, s)[0], M.score_counts(nB, s)[0]
            bs = M.paired_bootstrap_counts(nB, nRef, site_family, BOOT, 0, s) if BOOT else {}
            sc[k] = {"A": sa, "B": sb, "delta": sb - sa, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
        pa, pb = M.score_counts(nRef)[1], M.score_counts(nB)[1]
        fl = {c: {"f1_A": pa[c]["f1"], "f1_B": pb[c]["f1"], "d": pb[c]["f1"] - pa[c]["f1"], "n_gt": pa[c]["n_gt"], "floor": FLOOR.get(c, -0.005),
                  "ok": (pb[c]["f1"] - pa[c]["f1"] >= FLOOR.get(c, -0.005) - 1e-12) or pa[c]["n_gt"] < 5,
                  "tp": [pa[c]["tp"], pb[c]["tp"]], "fp": [pa[c]["fp"], pb[c]["fp"]]} for c in pa if pa[c]["n_gt"] > 0}
        g = {"delta_ge_0.003": sc["all"]["delta"] >= 0.003, f"P_ge_{P_MIN}": (sc["all"]["P"] or 0) >= P_MIN,
             "class_floors": all(v["ok"] for v in fl.values()),
             "new34_ge_0": sc["new34"]["delta"] >= -1e-9, "nolidar_ge_0": sc["nolidar"]["delta"] >= -1e-9}
        g["PASS"] = all(g.values()); g["info_P_ge_0.975"] = (sc["all"]["P"] or 0) >= 0.975
        res["variants"][name] = {"decisive": gate, "n_rows": n_rows, "scores": sc, "floor": fl, "gate": g}
        log(f"{name}: A {sc['all']['A']:.6f} B {sc['all']['B']:.6f} Δ {sc['all']['delta']:+.6f} P {sc['all']['P']} | new34 {sc['new34']['delta']:+.6f} "
            f"| nolidar {sc['nolidar']['delta']:+.6f} | floors {{{', '.join(f'{c[:6]} {v['d']:+.4f}' for c, v in fl.items())}}} | PASS {g['PASS']}"); save()

    nodes = {}
    for name, paths in VAR.items():
        rows = []
        for p in paths:
            for f, _ in D.iter_features(p):
                q = f["properties"]
                if q.get("class_name") not in D.CLASSES5: continue
                q["threshold_group"] = "li_cnxblend" if name != "ref_s2_single" else "li_cnx"; q["markup_type"] = "li"; rows.append(f)
        rB = fl_feats(rows)
        nr = {"all": len(rows), "ctx": len(rB), "classes_ctx": dict(Counter(f["properties"]["class_name"] for f in rB))}
        cB, _ = bctx.curves(ex_ctx + rB); nB = D.honest(cB, folds)[0]; del cB, rows, rB; gc.collect()
        nodes[name] = nB
        score(name, nB, nA, gate=name in DECISIVE, n_rows=nr)
    for name in DECISIVE:   # report only: vs single seed s2 in the same context
        score(f"info_{name}_vs_s2_single", nodes[name], nodes["ref_s2_single"])
    res["sec"] = round(time.perf_counter() - t0); save(); log("done", res["sec"], "s")

if __name__ == "__main__":
    main()
