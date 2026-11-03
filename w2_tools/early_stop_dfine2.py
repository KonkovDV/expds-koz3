#!/usr/bin/env python3
"""Early stop of P-W2-13 (D-FINE-2), run when kernel dmitrijkv/koz3-w2-dfine2-a (folds 0,3) has landed. This is not the gate.
usage: python w2_tools/early_stop_dfine2.py <output dir of koz3-w2-dfine2-a> [--folds 0,3] [--out json]
Needs oof_dfine_f{k}.geojson for every fold k. The reference is the existing sp_dfine OOF for the same folds,
taken from /workspace/koz3/w4geo/spor_oof/{full,rest}/oof_dfine_spor.geojson.
Satellite standalone score on the fold regions: compute_metrics against GT gt_train_v2 sp_or, with every
prediction set turned into circles first.
  A = existing sp_dfine. Circles via cluster_circles(radius 4 m, merge 4 m), same as prod / W4 dfine_feats.
  B = D-FINE-2, with the same circles.
  M = A+B merged. cluster_circles(radius 4 m, merge 8 m, ensemble mean, n_models 2), the gate's merge rule.
  C = control: A merged with a copy of itself by the same rule (what a useless clone of A would give; the 8 m merge
      alone already moves S, so the gain of M is measured against C, not against A).
S(X) is the CLASS_WEIGHTS-weighted mean over the classes present in GT of max_t F1_c(conf >= t).
The grid is t = 0.05..0.95, step 0.025, and it is the same for every set.
Rules (PREREG P-W2-13):
  R1: S(B) < S(A) - 0.02 -> FAIL (the member is clearly weaker).
  R2: S(M) <= S(C) -> FAIL (the second member adds nothing beyond the 8 m merge itself on f0+f3).
On FAIL: cancel koz3-w2-dfine2-b and release the leases. On PASS: wait for b, then run the gate."""
import argparse, json, logging, os, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "materials"))
import compute_metrics as CM
from expds.sp_dfine import cluster_circles
from shapely.geometry import shape, mapping
logging.getLogger("compute_metrics").setLevel(logging.ERROR)
REF = [Path("/workspace/koz3/w4geo/spor_oof/full/oof_dfine_spor.geojson"), Path("/workspace/koz3/w4geo/spor_oof/rest/oof_dfine_spor.geojson")]
GT = Path("/workspace/koz3/ds/gt/gt_train_v2.geojson")
GRID = [round(0.05 + 0.025 * i, 3) for i in range(37)]
CLS5 = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]


def dets(paths, folds, model):
    out = []
    for p in paths:
        for f in json.load(open(p, encoding="utf-8"))["features"]:
            q = f["properties"]
            if int(q.get("fold", -1)) in folds and q.get("class_name") in CLS5:
                out.append({"geom": shape(f["geometry"]), "class_name": q["class_name"], "confidence": float(q["confidence"]),
                            "region_name": q["region_name"], "model": model})
    return out


def fc(feats):
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": feats}


def score(circles, gt, regions):
    feats = [{"type": "Feature", "geometry": mapping(c["geom"]), "properties": {"class_name": c["class_name"], "confidence": float(c["confidence"]),
              "region_name": c["region_name"]}} for c in circles if c["region_name"] in regions]
    present = sorted({f["properties"]["class_name"] for f in gt["features"]})
    best = {c: {"f1": 0.0, "t": None} for c in present}
    for t in GRID:
        r = CM.F1Calculator().calculate_f1(fc([f for f in feats if f["properties"]["confidence"] >= t]), gt)
        for c in present:
            v = r["per_class"].get(c, {})
            if v.get("f1", 0.0) > best[c]["f1"]:
                best[c] = {"f1": v["f1"], "t": t, "tp": v.get("tp"), "fp": v.get("fp"), "fn": v.get("fn")}
    w = CM.CLASS_WEIGHTS
    s = sum(best[c]["f1"] * w[c] for c in present) / sum(w[c] for c in present)
    return {"S": s, "per_class": best, "n_circles": len(feats)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("--folds", default="0,3"); ap.add_argument("--out")
    ap.add_argument("--ref", nargs="*", default=[str(p) for p in REF], help="reference OOF (sp_dfine); default W4 spor_oof")
    a = ap.parse_args(); folds = {int(k) for k in a.folds.split(",")}
    d2p = [Path(a.d) / f"oof_dfine_f{k}.geojson" for k in sorted(folds)]
    miss = [str(p) for p in d2p if not p.exists()]
    assert not miss, f"missing {miss}"
    A0, B0 = dets(map(Path, a.ref), folds, 0), dets(d2p, folds, 1)
    regs = {d["region_name"] for d in A0}; regs_b = {d["region_name"] for d in B0}
    assert regs_b <= regs or not regs_b, f"D-FINE-2 regions outside the reference folds: {sorted(regs_b - regs)[:5]}"
    gtall = json.load(open(GT, encoding="utf-8"))
    gt = fc([f for f in gtall["features"] if f["properties"].get("markup_type") == "sp_or" and f["properties"].get("region_name") in regs
             and f["properties"].get("class_name") in CLS5])
    A = cluster_circles(A0, 4.0, 4.0); B = cluster_circles(B0, 4.0, 4.0)
    Mg = cluster_circles(A0 + B0, 4.0, 8.0, "mean", 2)
    Cg = cluster_circles(A0 + [dict(d, model=1) for d in A0], 4.0, 8.0, "mean", 2)
    res = {"folds": sorted(folds), "regions": len(regs), "regions_b": len(regs_b), "n_gt": len(gt["features"]),
           "gt_classes": dict(Counter(f["properties"]["class_name"] for f in gt["features"])),
           "A_sp_dfine": score(A, gt, regs), "B_dfine2": score(B, gt, regs), "M_merge8_mean": score(Mg, gt, regs),
           "C_self_merge8_mean": score(Cg, gt, regs)}
    sA, sB, sM = res["A_sp_dfine"]["S"], res["B_dfine2"]["S"], res["M_merge8_mean"]["S"]
    sC = res["C_self_merge8_mean"]["S"]
    res["R1_fail"] = sB < sA - 0.02; res["R2_fail"] = sM <= sC
    res["dB"], res["dM"], res["dM_vs_C"] = sB - sA, sM - sA, sM - sC
    res["verdict"] = "FAIL" if (res["R1_fail"] or res["R2_fail"]) else "PASS"
    s = json.dumps(res, ensure_ascii=False, indent=1, default=float)
    if a.out: Path(a.out).write_text(s, encoding="utf-8")
    print(s)


if __name__ == "__main__":
    main()
