#!/usr/bin/env python3
"""Прокси-ворота W4 P-clean (docs/PREREG_w4_pix_clean.md §2): A (классика) против C (pix_clean, replace).

    python scripts/pix_clean_eval.py --gt GT.geojson --a A.geojson --c C_s0.geojson C_s1.geojson ... --out R.json

Каждый C — объединение OOF-предсказаний по фолдам F1–F5 (модель фолда не видела его участков).
Пороги — honest_counts на тех же фолдах для A и C. Бутстреп — по семьям (и справочно по участкам).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import metric  # noqa: E402
from expds.constants import CLASS_WEIGHTS  # noqa: E402
from expds.naming import site_family  # noqa: E402

CLASSES = ("finds_points", "object_poly")
FOLDS = {"F1": ["200"], "F2": [str(i) for i in range(201, 208)], "F3": ["208", "209", "210", "215"],
         "F4": ["211", "212", "216"], "F5": ["213", "214", "217"]}


def _load(path, regions=None, mt=None):
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = []
    for f in fc["features"]:
        pr = f.get("properties") or {}
        if pr.get("class_name") not in CLASSES:
            continue
        if regions is not None and pr.get("region_name") not in regions:
            continue
        if mt is not None and pr.get("markup_type") != mt:
            continue
        feats.append(f)
    return {**{k: v for k, v in fc.items() if k != "features"}, "features": feats}


def folds_of(regions: set[str]) -> dict[str, set[str]]:
    return {k: {r for r in regions if r.split("_")[0] in nums} for k, nums in FOLDS.items()}


def honest(pred, gt, folds):
    curves = metric.build_curves(pred, gt)
    counts, info = metric.honest_counts(curves, [s for s in folds.values() if s])
    return counts, info


def summary(counts, folds):
    s, per = metric.score_counts(counts)
    out = {"score2": s, "f1": {c: per[c]["f1"] for c in per if c in CLASSES},
           "folds": {k: metric.score_counts(counts, v)[0] for k, v in folds.items() if v}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--c", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    gt = _load(args.gt)
    regions = {f["properties"]["region_name"] for f in gt["features"]}
    gt = _load(args.gt, regions)
    folds = folds_of(regions)
    w_all = sum(CLASS_WEIGHTS.values())
    w2 = sum(CLASS_WEIGHTS.get(c, 0.0) for c in CLASSES)
    scale = w2 / w_all                      # балл по двум классам → вклад в общий балл

    a_counts, a_info = honest(_load(args.a, regions), gt, folds)
    res = {"regions": sorted(regions), "folds": {k: sorted(v) for k, v in folds.items()},
           "scale_to_global": scale, "A": summary(a_counts, folds), "A_info": a_info, "C": []}
    for path in args.c:
        c_counts, c_info = honest(_load(path, regions), gt, folds)
        sa, sc = res["A"], summary(c_counts, folds)
        d = {"file": Path(path).name, "summary": sc, "info": c_info,
             "d_f1": {c: sc["f1"].get(c, 0.0) - sa["f1"].get(c, 0.0) for c in CLASSES},
             "d_global": (sc["score2"] - sa["score2"]) * scale,
             "d_folds_global": {k: (sc["folds"][k] - sa["folds"][k]) * scale for k in sc["folds"]},
             "boot_family": metric.paired_bootstrap_counts(c_counts, a_counts, site_family, args.boot, 0),
             "boot_region": metric.paired_bootstrap_counts(c_counts, a_counts, lambda r: r, args.boot, 0)}
        for b in ("boot_family", "boot_region"):
            for k in ("mean", "p5", "p50", "p95"):
                if k in d[b]:
                    d[b][k + "_global"] = d[b][k] * scale
        slices = {}
        for mt in ("maggr", "etgr", "terfl"):
            gm = _load(args.gt, regions, mt)
            if not gm["features"]:
                continue
            ca, _ = honest(_load(args.a, regions, mt), gm, folds)
            cc, _ = honest(_load(path, regions, mt), gm, folds)
            slices[mt] = {"A": metric.score_counts(ca)[0] * scale, "C": metric.score_counts(cc)[0] * scale,
                          "d_global": (metric.score_counts(cc)[0] - metric.score_counts(ca)[0]) * scale}
        d["slices_method"] = slices
        n_pos = sum(1 for v in d["d_folds_global"].values() if v > 0)
        d["gate_proxy"] = {
            "d_ge_0.009": d["d_global"] >= 0.009,
            "p_family_ge_0.95": d["boot_family"].get("p_gt0", 0.0) >= 0.95,
            "folds_pos": f"{n_pos}/{len(d['d_folds_global'])}",
            "folds_ge_4": n_pos >= 4,
            "methods_ge_-0.003": all(v["d_global"] >= -0.003 for v in slices.values()),
            "f1_not_below_A": all(v >= 0 for v in d["d_f1"].values()),
        }
        d["gate_proxy"]["accepted"] = all(v for k, v in d["gate_proxy"].items() if k != "folds_pos")
        res["C"].append(d)
        print(json.dumps({k: d[k] for k in ("file", "d_f1", "d_global", "d_folds_global", "gate_proxy")},
                         ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
