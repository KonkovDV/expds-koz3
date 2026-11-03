#!/usr/bin/env python3
"""P-W2-13 gate: D-FINE-2 (second sp_or D-FINE member) merged with the existing sp_dfine within 8 m, on V3P in the
v3_on PROD CONTEXT (output_floors / li_floor of the v3_on archive config, same as P-W2-9 amendment 2 / r50c5).
Engine = W4 v3_gate_lib (d3_gate TableEngine, honest-10 family folds, paired bootstrap BOOT by site_family, seed 0).
A = V3P_ctx (c19 + sp_dfine as in V3P). B = same, sp_dfine replaced by ENS8 = per region
    SPD.cluster_circles(ref(model 0) + dfine2(model 1), radius 4 m, merge 8 m, ensemble "mean", n_models 2)
    -> P.finalize(P.merge_within_markup(.)), threshold_group "sp_dfine_ens" (no archive section -> archive class defaults).
PASS (all): Δall >= +0.003, P(Δ>0) >= 0.95, Δnew34 >= 0, Δnolidar >= 0,
            per-class ΔF1 >= -0.002 for kp/kt/gor and >= -0.005 for arkh/fort (classes with n_gt < 5 exempt, as r50c5).
Run: nice -n 10 /workspace/koz3/w4venv/bin/python gate_dfine2_ctx.py [BOOT] [OUT] [D2_DIR ...]
     D2_DIR default: /workspace/koz3/w2/oof/dfine2/dmitrijkv_koz3-w2-dfine2-{a,b} (each holding oof_dfine_f{k}.geojson)."""
import importlib.util, json, sys, time, gc, tomllib, copy
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import shape, mapping
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
sys_argv = sys.argv; sys.argv = sys.argv[:1]
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G); sys.argv = sys_argv
D, M, P, SPD, log, site_family = G.D, G.M, G.P, G.SPD, G.log, G.site_family
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "gate_dfine2_ctx.json"
R = Path("/workspace/koz3/w2/oof/dfine2")
D2_DIRS = [Path(p) for p in sys.argv[3:]] or [R / "dmitrijkv_koz3-w2-dfine2-a", R / "dmitrijkv_koz3-w2-dfine2-b"]
V3P_REF = 0.3337867244
CFG_P = next(p for p in (HERE / "v3on_src/configs/default.toml", Path("/workspace/koz3/w2/r50c5_gate/v3on_src/configs/default.toml")) if p.exists())
# ^ extracted from w4geo/dist/koz3_v3on_b505656_6d4764f8_w6dfd2e07.zip (not committed); floors cross-checked against v3on_floors.json below
CLASS_FLOOR = {"kurgany_povrezhdennye": -0.002, "kurgany_tselye": -0.002, "gorodishcha": -0.002, "arkhitektury": -0.005, "fortifikatsii": -0.005}
cfg = tomllib.load(open(CFG_P, "rb"))
post = cfg.get("post") or {}
assert not post.get("kurgan_neighbour_prior") and not post.get("site_calibration") and "w5_cloudfeat" not in cfg
_FL = {}
def floor(g, c):   # == expds/pipeline.output_floors(cfg, g)[c] of v3_on (no prior/calib active)
    if g not in _FL:
        thr, own, dft = cfg.get("thresholds", {}), cfg.get("thresholds_by_markup", {}).get(g, {}), cfg.get("default_threshold", 0.15)
        _FL[g] = {cc: float(own.get(cc, thr.get(cc, dft))) for cc in D.CLASSES5 + ("object_poly", "finds_points")}
    return _FL[g].get(c, float(cfg.get("default_threshold", 0.15)))

def ctx_engine(tab):
    t2 = copy.copy(tab); t2.base = {}; n0 = n1 = 0
    for c, b in tab.base.items():
        keep = np.array([cf >= floor(g, c) for g, cf in zip(b["grp"], b["conf"])], bool)
        n0 += len(keep); n1 += int(keep.sum())
        idx = np.flatnonzero(keep); new = {int(o): n for n, o in enumerate(idx)}
        t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx], pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    return t2, n0, n1

def fl_feats(fs):
    return [f for f in fs if float(f["properties"]["confidence"]) >= floor(M.threshold_group(f["properties"]), f["properties"]["class_name"])]

def raw_dets(paths, model):   # same row construction as G.dfine_feats
    by = defaultdict(list); fold_reg = defaultdict(set)
    for p in paths:
        for f, _ in G.iter_features(p):
            q = f["properties"]
            if q.get("class_name") not in D.CLASSES5: continue
            fold_reg[q.get("region_name")].add(q.get("fold"))
            by[q["region_name"]].append({"geom": shape(f["geometry"]), "class_name": q["class_name"], "confidence": float(q["confidence"]),
                                         "region_name": q["region_name"], "source": q.get("tif"), "original_crs": q.get("original_crs"), "model": model})
    assert all(len(v) == 1 for v in fold_reg.values()), "region in several folds"
    return by, {r: next(iter(v)) for r, v in fold_reg.items()}

def circles(by_list, merge_m, ens, n_models, group):
    out = []
    for reg in sorted(set().union(*by_list)):
        dets = [d for by in by_list for d in by.get(reg, [])]
        for f in P.finalize(P.merge_within_markup(SPD.cluster_circles(dets, 4.0, merge_m, ens, n_models))):
            out.append({"type": "Feature", "geometry": mapping(f["geom"]),
                        "properties": {"class_name": f["class_name"], "confidence": round(float(f["confidence"]), 6), "region_name": f["region_name"],
                                       "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": f.get("original_crs"), "threshold_group": group}})
    return out

def main():
    t0 = time.perf_counter(); res = {"what": "P-W2-13: D-FINE-2 ens8 (mean) on V3P, v3_on prod context", "boot": BOOT, "cfg": str(CFG_P),
                                     "d2_dirs": [str(p) for p in D2_DIRS], "variants": {}}
    save = lambda: OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    d2p = sorted(p for d in D2_DIRS for p in d.glob("oof_dfine_f[0-4].geojson"))
    ref, ref_fold = raw_dets(G.OOF, 0); d2, d2_fold = raw_dets(d2p, 1)
    res["d2_files"] = [str(p) for p in d2p]; res["n_ref"] = sum(map(len, ref.values())); res["n_d2"] = sum(map(len, d2.values()))
    res["d2_regions"] = len(d2); res["ref_regions"] = len(ref)
    bad = {r: (ref_fold.get(r), f) for r, f in d2_fold.items() if ref_fold.get(r) != f}
    assert not bad, f"D-FINE-2 fold/region mismatch vs ref: {list(bad.items())[:5]}"
    res["d2_missing_regions"] = sorted(set(ref) - set(d2))
    assert sorted({ref_fold[r] for r in d2}) == [0, 1, 2, 3, 4] and not res["d2_missing_regions"], f"D-FINE-2 OOF incomplete: {res['d2_missing_regions'][:5]}"
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
    # S0: V3P exact; S1: this script's circle path on ref alone (merge 4, max, 1 model, group sp_dfine) reproduces V3P
    cP, _ = base3.curves(c19 + dfx); nP = D.honest(cP, folds)[0]; del cP
    res["V3P"] = M.score_counts(nP)[0]; res["S0_ok"] = abs(res["V3P"] - V3P_REF) <= 1e-6
    s1 = circles([ref], 4.0, "max", 1, "sp_dfine"); cS1, _ = base3.curves(c19 + s1); res["S1"] = M.score_counts(D.honest(cS1, folds)[0])[0]; del cS1
    res["S1_ok"] = abs(res["S1"] - V3P_REF) <= 1e-6
    log(f"S0 V3P {res['V3P']:.10f} ok {res['S0_ok']} | S1 {res['S1']:.10f} ok {res['S1_ok']}"); save()
    res["S0_note"] = "C19 geo eval missing in backups: V3P differs only in etgr object_poly; delta/P class-separable"
    assert abs(res["V3P"] - res["S1"]) <= 1e-9, "circle path != V3P"
    # A = V3P_ctx
    bctx, n0, n1 = ctx_engine(base3)
    c19c, dfxc = fl_feats(c19), fl_feats(dfx)
    res["ctx_rows"] = {"stack_before": n0, "stack_kept": n1, "c19_kept": len(c19c), "dfx_before": len(dfx), "dfx_kept": len(dfxc)}
    cA, _ = bctx.curves(c19c + dfxc); nA = D.honest(cA, folds)[0]; del cA
    res["V3P_ctx"] = M.score_counts(nA)[0]; log(f"V3P_ctx {res['V3P_ctx']:.10f} rows {res['ctx_rows']}"); save()

    def score(name, nB, nRef, gate=False):
        sc = {}
        for k, s in slices.items():
            sa, sb = M.score_counts(nRef, s)[0], M.score_counts(nB, s)[0]
            bs = M.paired_bootstrap_counts(nB, nRef, site_family, BOOT, 0, s) if BOOT else {}
            sc[k] = {"A": sa, "B": sb, "delta": sb - sa, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
        pa, pb = M.score_counts(nRef)[1], M.score_counts(nB)[1]
        fl = {c: {"f1_A": pa[c]["f1"], "f1_B": pb[c]["f1"], "d": pb[c]["f1"] - pa[c]["f1"], "n_gt": pa[c]["n_gt"], "floor": CLASS_FLOOR.get(c, -0.005),
                  "ok": (pb[c]["f1"] - pa[c]["f1"] >= CLASS_FLOOR.get(c, -0.005) - 1e-12) or pa[c]["n_gt"] < 5,
                  "tp": [pa[c]["tp"], pb[c]["tp"]], "fp": [pa[c]["fp"], pb[c]["fp"]]} for c in pa if pa[c]["n_gt"] > 0}
        g = {"delta_ge_0.003": sc["all"]["delta"] >= 0.003, "P_ge_0.95": (sc["all"]["P"] or 0) >= 0.95,
             "class_floors": all(v["ok"] for v in fl.values()),
             "new34_ge_0": sc["new34"]["delta"] >= 0, "nolidar_ge_0": sc["nolidar"]["delta"] >= 0}
        g["PASS"] = all(g.values())
        res["variants"][name] = {"decisive": gate, "scores": sc, "floor": fl, "gate": g}
        log(f"{name}: A {sc['all']['A']:.6f} B {sc['all']['B']:.6f} Δ {sc['all']['delta']:+.6f} P {sc['all']['P']} | new34 {sc['new34']['delta']:+.6f} "
            f"| nolidar {sc['nolidar']['delta']:+.6f} | floors {{{', '.join(f'{c[:6]} {v['d']:+.4f}' for c, v in fl.items())}}} | PASS {g['PASS']}"); save()

    def variant(name, merge_m, ens, group, gate=False, vs_raw=False):
        e = circles([ref, d2], merge_m, ens, 2, group); ec = fl_feats(e)
        res.setdefault("ens_rows", {})[name] = {"before": len(e), "kept": len(ec), "classes": dict(Counter(f["properties"]["class_name"] for f in ec))}
        cB, _ = bctx.curves(c19c + ec); nB = D.honest(cB, folds)[0]; del cB
        score(name, nB, nA, gate)
        if vs_raw: score(name + "_vs_V3P", nB, nP)
        return e

    import os
    ENS_CLS = set(os.environ.get("ENS_CLS", "kurgany_povrezhdennye,gorodishcha,arkhitektury").split(","))   # PREREG_cls_20261103.md

    def variant_cls(name, ens_group, p_need):
        e = [f for f in circles([ref, d2], 8.0, "mean", 2, ens_group) if f["properties"]["class_name"] in ENS_CLS]
        r = [f for f in circles([ref], 4.0, "max", 1, "sp_dfine") if f["properties"]["class_name"] not in ENS_CLS]
        ec = fl_feats(e + r)
        res.setdefault("ens_rows", {})[name] = {"ens": len(e), "ref": len(r), "kept": len(ec),
                                                "classes": dict(Counter(f["properties"]["class_name"] for f in ec))}
        cB, _ = bctx.curves(c19c + ec); nB = D.honest(cB, folds)[0]; del cB
        score(name, nB, nA, True)
        g = res["variants"][name]["gate"]; g["P_ge_bonferroni"] = (res["variants"][name]["scores"]["all"]["P"] or 0) >= p_need
        g["PASS"] = g["PASS"] and g["P_ge_bonferroni"]; g["p_need"] = p_need
        log(f"{name}: PASS(prereg, P>={p_need}) {g['PASS']}"); save()

    if os.environ.get("ONLY_CLS") == "1":
        if os.environ.get("ONLY_C3") == "1":
            variant_cls("C3_ens8_arkh_gor_as_sp_dfine_ctx", "sp_dfine", 0.9875)
        else:
            variant_cls("C_ens8_cls_ctx_vs_V3P_ctx", "sp_dfine_ens", 0.9833)
            variant_cls("C2_ens8_cls_as_sp_dfine_ctx", "sp_dfine", 0.9833)
    else:
        variant("B_ens8_mean_ctx_vs_V3P_ctx", 8.0, "mean", "sp_dfine_ens", gate=True, vs_raw=True)   # decisive
        variant("ens8_max_ctx", 8.0, "max", "sp_dfine_ens")                                         # report-only
        variant("ens4_mean_ctx", 4.0, "mean", "sp_dfine_ens")                                       # report-only
        variant("ens8_mean_as_sp_dfine_ctx", 8.0, "mean", "sp_dfine")                               # report-only: archive sp_dfine floors
    res["floors_used"] = _FL
    ref_fl = json.load(open(HERE / "v3on_floors.json"))["floors"]
    res["floors_match_v3on_floors_json"] = all(abs(_FL[g][c] - float(v)) < 1e-12 for g, d in ref_fl.items() if isinstance(d, dict) and g in _FL
                                               for c, v in d.items() if c in _FL[g])
    res["sec"] = round(time.perf_counter() - t0); save(); log("done", res["sec"], "s")

if __name__ == "__main__":
    main()
