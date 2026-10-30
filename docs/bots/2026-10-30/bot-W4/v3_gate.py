#!/usr/bin/env python3
"""W4 v3_on gate (PREREG docs/PREREG_w4_v3on.md): B0=v2clean vs V3P/V3F, honest-10, d3_gate machinery imported unchanged."""
import importlib.util, json, sys, time, gc
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("d3_gate", "/workspace/koz3/orch/bot-W1/work/d3_gate.py")
D = importlib.util.module_from_spec(_s); sys.modules["d3_gate"] = D; _s.loader.exec_module(D)
M, P, S, SPD, log, site_family, iter_features = D.M, D.P, D.S, D.SPD, D.log, D.site_family, D.iter_features
from shapely.geometry import shape, mapping

BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/workspace/koz3/orch/bot-W4/work/v3_gate.json")
SP = {"sp_or", "sp_focal", "sp_arch", "sp_focal2"}
OOF = [Path("/workspace/koz3/w4geo/spor_oof/full/oof_dfine_spor.geojson"), Path("/workspace/koz3/w4geo/spor_oof/rest/oof_dfine_spor.geojson")]
TAU = {"kurgany_povrezhdennye": 0.555265, "kurgany_tselye": 0.670285, "gorodishcha": 0.68847, "arkhitektury": 0.373015, "fortifikatsii": 0.869051}


def load_geo(p, pred):
    return [f for f, _ in iter_features(Path(p)) if pred(f["properties"])]


def drop_engine(tab, drop):   # copy of TableEngine base without rows where drop(grp, reg)
    import copy
    t2 = copy.copy(tab); t2.base = {}
    for c, b in tab.base.items():
        keep = np.array([not drop(g, r) for g, r in zip(b["grp"], b["reg"])], bool)
        idx = np.flatnonzero(keep); new = {int(o): n for n, o in enumerate(idx)}
        t2.base[c] = dict(conf=b["conf"][idx], reg=b["reg"][idx], grp=b["grp"][idx],
                          pairs={new[i]: v for i, v in b["pairs"].items() if i in new})
    return t2


def dfine_feats():
    by = defaultdict(list); fold_reg = defaultdict(set); n = 0
    for p in OOF:
        for f, _ in iter_features(p):
            q = f["properties"]
            if q.get("class_name") not in D.CLASSES5:
                continue
            n += 1; fold_reg[q.get("region_name")].add(q.get("fold"))
            by[q["region_name"]].append({"geom": shape(f["geometry"]), "class_name": q["class_name"], "confidence": float(q["confidence"]),
                                         "region_name": q["region_name"], "source": q.get("tif"), "original_crs": q.get("original_crs"), "model": 0})
    assert all(len(v) == 1 for v in fold_reg.values()), "region in several folds"
    out = []
    for reg in sorted(by):
        cl = SPD.cluster_circles(by[reg], 4.0, 4.0)
        for f in P.finalize(P.merge_within_markup(cl)):
            out.append({"type": "Feature", "geometry": mapping(f["geom"]),
                        "properties": {"class_name": f["class_name"], "confidence": round(float(f["confidence"]), 6), "region_name": f["region_name"],
                                       "crs": "EPSG:3857", "markup_type": "sp_or", "original_crs": f.get("original_crs"), "threshold_group": "sp_dfine"}})
        by[reg] = None
    log(f"D-FINE OOF: {n} ellipses, {len(fold_reg)} regions -> {len(out)} circles {dict(Counter(f['properties']['class_name'] for f in out))}")
    return out, set(fold_reg)


def main():
    t0 = time.perf_counter()
    stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
                  Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
    tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
    slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
    cA, _ = tab.curves()
    regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
    folds = M.family_folds(regs, site_family, 10)
    nA, _, _ = D.honest(cA, folds); res = {"A_honest": M.score_counts(nA)[0], "boot": BOOT, "variants": {}}
    log(f"A honest {res['A_honest']:.10f}")
    no_et = drop_engine(tab, lambda g, r: g == "etgr")
    a18 = load_geo("/workspace/koz3/w4geo/eval/A18.geojson", lambda q: q.get("markup_type") == "etgr")
    c19 = load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
    s1, _ = no_et.curves(a18); n1, _, _ = D.honest(s1, folds)
    res["S1_A_rebuilt_from_A18"] = M.score_counts(n1)[0]
    log(f"S1 A via A18 extras {res['S1_A_rebuilt_from_A18']:.10f} (must = A) a18 {len(a18)} c19 {len(c19)}")
    cB0, _ = no_et.curves(c19); nB0, _, _ = D.honest(cB0, folds)
    res["B0_honest"] = {k: M.score_counts(nB0, s)[0] for k, s in slices.items()}
    res["B0_per_class"] = M.score_counts(nB0)[1]
    log(f"B0 v2clean honest {res['B0_honest']}")
    dfx, dreg = dfine_feats()
    base3 = drop_engine(tab, lambda g, r: g == "etgr" or (g in SP and r in dreg))
    res["dropped_sp_rows"] = sum(len(b["conf"]) for b in no_et.base.values()) - sum(len(b["conf"]) for b in base3.base.values())
    res["sp_rows_left"] = int(sum(np.isin(b["grp"], list(SP)).sum() for b in base3.base.values()))
    log(f"dropped sp rows {res['dropped_sp_rows']}, sp rows left {res['sp_rows_left']}")
    del tab; gc.collect()

    def score(name, curves, fixed=frozenset()):
        rB = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
        assert rB <= regs, "new regions"
        nB, info, _ = D.honest(curves, folds, fixed=set(fixed))
        sc = {}
        for k, s in slices.items():
            sa, sb = M.score_counts(nB0, s)[0], M.score_counts(nB, s)[0]
            bs = M.paired_bootstrap_counts(nB, nB0, site_family, BOOT, 0, s) if BOOT else {}
            sc[k] = {"B0": sa, "B": sb, "delta": sb - sa, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
        pa, pb = M.score_counts(nB0)[1], M.score_counts(nB)[1]
        floor = {c: {"f1_B0": pa[c]["f1"], "f1_B": pb[c]["f1"], "d": pb[c]["f1"] - pa[c]["f1"],
                     "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12,
                     "tp": [pa[c]["tp"], pb[c]["tp"]], "fp": [pa[c]["fp"], pb[c]["fp"]]} for c in pa if pa[c]["n_gt"] > 0}
        g = {"delta_ge_0.003": sc["all"]["delta"] >= 0.003, "P_ge_0.9": (sc["all"]["P"] or 0) >= 0.9,
             "floor_ok": all(v["ok"] for v in floor.values()), "new34_nolidar_not_down": sc["new34"]["delta"] >= 0 and sc["nolidar"]["delta"] >= 0}
        g["PASS"] = all(g.values())
        res["variants"][name] = {"scores": sc, "floor": floor, "gate": g, "unseen_dropped": info["unseen_dropped"]}
        log(f"{name}: B {sc['all']['B']:.6f} Δall {sc['all']['delta']:+.6f} P {sc['all']['P']} | new34 {sc['new34']['delta']:+.6f} "
            f"| nolidar {sc['nolidar']['delta']:+.6f} | floor {g['floor_ok']} | PASS {g['PASS']}")
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

    score("S2_B0_vs_B0", cB0)
    cP, _ = base3.curves(c19 + dfx); score("V3P", cP); del cP; gc.collect()
    dfF = [f for f in dfx if f["properties"]["confidence"] >= TAU[f["properties"]["class_name"]]]
    cF, _ = base3.curves(c19 + dfF); score("V3F", cF, fixed={"sp_dfine"})
    res["n_dfine"] = {"all": len(dfx), "fixed": len(dfF)}; res["sec"] = round(time.perf_counter() - t0)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str)); log("done", res["sec"], "s")


main()
