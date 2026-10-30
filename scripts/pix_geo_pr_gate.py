#!/usr/bin/env python3
"""Geo P/R gate for PIXGBM recover (orch/PIXGBM_RECOVER_20261030.md §4–5).

Compares candidate C* to a reference REF (v2-ref or A18) on the geo5 pool
(regions 200–217) using the official compute_metrics modality block (same
formula as LB scoring.log «геофизика precision/recall»).

    python scripts/pix_geo_pr_gate.py \\
        --gt /workspace/koz3/w5data/gt_train_v2.geojson \\
        --c  C_star.geojson \\
        --ref REF.geojson \\
        --out runs/geo_pr_gate.json

Optional --a A18.geojson adds honest score2 Δ via pix_clean_eval (surrogate
when v2-ref missing). Gate PASS when --ref given and C geo P≥REF.P and
R≥REF.R (targets for v2-ref: ≥0.769 / ≥0.656). Without --ref: prints C
metrics only (not Elena-ready).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "materials"))

from compute_metrics import F1Calculator  # noqa: E402
import pix_clean_eval as ev  # noqa: E402

GEO_NUMS = {str(i) for i in range(200, 218)}
GEO_METHODS = ("maggr", "etgr", "terfl")
V2_GEO_P, V2_GEO_R = 0.769418, 0.656489  # LB 29.10 bot7cand_v2


def _filter_fc(path: str, regions: set[str] | None = None, mt: str | None = None) -> dict:
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = []
    for f in fc.get("features") or []:
        pr = f.get("properties") or {}
        rn = pr.get("region_name") or ""
        if regions is not None and rn not in regions:
            continue
        if mt is not None and pr.get("markup_type") != mt:
            continue
        feats.append(f)
    return {**{k: v for k, v in fc.items() if k != "features"}, "features": feats}


def _geo_regions(gt_path: str) -> set[str]:
    fc = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    return {
        (f.get("properties") or {}).get("region_name", "")
        for f in fc.get("features") or []
        if ((f.get("properties") or {}).get("region_name") or "").split("_")[0] in GEO_NUMS
    }


def _score(pred: dict, gt: dict) -> dict:
    calc = F1Calculator()
    return calc.calculate_f1(pred, gt)


def _geo_pr(results: dict) -> dict:
    aux = results.get("auxiliary") or {}
    dt = (results.get("by_data_type") or aux.get("by_data_type") or {})
    # Russian label from DATA_TYPE_GROUPS
    g = dt.get("геофизика") or dt.get("geophysics") or {}
    return {
        "precision": float(g.get("precision", 0.0)),
        "recall": float(g.get("recall", 0.0)),
        "tp": g.get("tp"), "fp": g.get("fp"), "fn": g.get("fn"),
    }


def _per_method(pred_path: str, gt_path: str, regions: set[str]) -> dict:
    out = {}
    gt_all = _filter_fc(gt_path, regions)
    for mt in GEO_METHODS:
        gt_m = _filter_fc(gt_path, regions, mt)
        if not gt_m["features"]:
            continue
        pr_m = _filter_fc(pred_path, regions, mt)
        # score against method-filtered GT; preds of other methods ignored by filter
        r = _score(pr_m, gt_m)
        # class F1 from by_class if present
        by_c = r.get("per_class") or r.get("by_class") or {}
        out[mt] = {
            "n_pred": len(pr_m["features"]),
            "n_gt": len(gt_m["features"]),
            "final_score": r.get("final_score"),
            "classes": {
                c: {"f1": (by_c.get(c) or {}).get("f1"),
                    "precision": (by_c.get(c) or {}).get("precision"),
                    "recall": (by_c.get(c) or {}).get("recall")}
                for c in ("finds_points", "object_poly")
                if c in by_c
            },
        }
        # also method-as-only-modality: rebuild using full scorer on filtered
        # (by_data_type will be single group)
        g = _geo_pr(r)
        out[mt]["precision"] = g["precision"]
        out[mt]["recall"] = g["recall"]
    return out


def _fmt_line(name: str, pr: dict) -> str:
    return f"  {name:10s} precision {pr['precision']:.6f}   recall {pr['recall']:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--c", required=True, help="candidate C* geojson")
    ap.add_argument("--ref", default=None, help="reference geojson (v2-ref preferred; else A18 surrogate)")
    ap.add_argument("--a", default=None, help="optional A18 for pix_clean_eval honest Δ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--v2-p", type=float, default=V2_GEO_P)
    ap.add_argument("--v2-r", type=float, default=V2_GEO_R)
    args = ap.parse_args()

    regions = _geo_regions(args.gt)
    gt = _filter_fc(args.gt, regions)
    c_fc = _filter_fc(args.c, regions)
    c_res = _score(c_fc, gt)
    c_pr = _geo_pr(c_res)
    c_methods = _per_method(args.c, args.gt, regions)

    print("===== GEO P/R GATE (geo5 pool, official modality) =====")
    print(f"regions: {len(regions)}")
    print(_fmt_line("C*", c_pr))
    for mt, m in c_methods.items():
        print(_fmt_line(f"C*.{mt}", m))

    ref_pr = ref_methods = None
    if args.ref:
        r_fc = _filter_fc(args.ref, regions)
        r_res = _score(r_fc, gt)
        ref_pr = _geo_pr(r_res)
        ref_methods = _per_method(args.ref, args.gt, regions)
        print(_fmt_line("REF", ref_pr))
        for mt, m in ref_methods.items():
            print(_fmt_line(f"REF.{mt}", m))

    print(_fmt_line("v2_LB", {"precision": args.v2_p, "recall": args.v2_r}))

    gate = {
        "has_ref": bool(args.ref),
        "c_ge_v2_p": c_pr["precision"] >= args.v2_p,
        "c_ge_v2_r": c_pr["recall"] >= args.v2_r,
    }
    if ref_pr:
        gate["c_ge_ref_p"] = c_pr["precision"] >= ref_pr["precision"]
        gate["c_ge_ref_r"] = c_pr["recall"] >= ref_pr["recall"]
        # method floors: no method < −0.003 recall vs REF
        method_ok = {}
        for mt in GEO_METHODS:
            if mt not in c_methods or mt not in (ref_methods or {}):
                continue
            d = c_methods[mt]["recall"] - ref_methods[mt]["recall"]
            method_ok[mt] = d >= -0.003
        gate["methods_ge_ref_minus_0.003"] = all(method_ok.values()) if method_ok else True
        gate["method_deltas_r"] = {
            mt: c_methods[mt]["recall"] - ref_methods[mt]["recall"]
            for mt in method_ok
        }
        gate["PASS_vs_ref"] = gate["c_ge_ref_p"] and gate["c_ge_ref_r"] and gate["methods_ge_ref_minus_0.003"]
        gate["PASS_elena_ready"] = gate["PASS_vs_ref"] and gate["c_ge_v2_p"] and gate["c_ge_v2_r"]
    else:
        gate["PASS_vs_ref"] = False
        gate["PASS_elena_ready"] = False
        gate["note"] = "no --ref: not Elena-ready (need v2-ref geojson)"

    honest = None
    if args.a:
        # surrogate: C vs A on geo classes via pix_clean_eval
        folds = ev.folds_of(regions)
        ac, _ = ev.honest(ev._load(args.a, regions), ev._load(args.gt, regions), folds)
        cc, _ = ev.honest(ev._load(args.c, regions), ev._load(args.gt, regions), folds)
        sa, sc = ev.summary(ac, folds), ev.summary(cc, folds)
        scale = sum(ev.CLASS_WEIGHTS.get(c, 0.0) for c in ev.CLASSES) / sum(ev.CLASS_WEIGHTS.values())
        d_global = (sc["score2"] - sa["score2"]) * scale
        honest = {"d_global": d_global, "d_f1": {c: sc["f1"].get(c, 0.0) - sa["f1"].get(c, 0.0) for c in ev.CLASSES},
                  "surrogate_ge_0.012": d_global >= 0.012}
        gate["surrogate_honest_ge_0.012"] = honest["surrogate_ge_0.012"]
        print(f"  honest Δ vs A (geo classes→global): {d_global:+.6f}  surrogate≥0.012={honest['surrogate_ge_0.012']}")

    res = {
        "gt": args.gt, "c": args.c, "ref": args.ref, "a": args.a,
        "n_regions": len(regions), "regions": sorted(regions),
        "C": {"geo": c_pr, "methods": c_methods, "final_score": c_res.get("final_score")},
        "REF": {"geo": ref_pr, "methods": ref_methods} if ref_pr else None,
        "v2_lb_target": {"precision": args.v2_p, "recall": args.v2_r},
        "honest_vs_A": honest,
        "gate": gate,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("gate", json.dumps(gate, ensure_ascii=False))
    return 0 if gate.get("PASS_elena_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
