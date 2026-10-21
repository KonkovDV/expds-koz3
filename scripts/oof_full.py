#!/usr/bin/env python3
"""Полный балл сдачи вне обучения: лидар, копии аэрофото, городища, геофизика.

Фортификации аэрофото в этот файл не входят, пока нет их предсказаний вне обучения.
Сырой лидар не прогоняется через слияние полигонов заново: так же считались 0,218,
и прирост от геофизики читается напрямую.

    python scripts/oof_full.py --work F:/expds3/work --metric C:/Sloi/files/materials/compute_metrics.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(path: Path) -> dict:
    print(f"читаю {path.name}", flush=True)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def names(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def keep_regions(fc: dict, regions: set[str] | None) -> dict:
    if regions is None:
        return fc
    feats = [f for f in fc["features"] if (f.get("properties") or {}).get("region_name") in regions]
    return {"type": "FeatureCollection", "features": feats}


def report(name: str, fc: dict, gt: dict, metric: str) -> float:
    from expds.metric import apply_group_thresholds, build_curves, official_weighted, score, sweep_groups

    print(f"{name}: объектов {len(fc['features'])}", flush=True)
    curves = build_curves(fc, gt, metric)
    res = sweep_groups(curves)
    taus: dict[str, dict[str, float]] = {}
    for cls, v in res.items():
        if v["f1"] <= 0 and not any(g["kept"] for g in v["groups"].values()):
            continue
        print(f"  {cls:24s} GT {v['n_gt']:5d}  F1 {v['f1']:.4f}", flush=True)
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
            if gv["kept"]:
                print(f"    {g:10s} порог {gv['threshold']:.4f}  оставлено {gv['kept']:6d}  TP {gv['tp']:5d}",
                      flush=True)
    est = official_weighted(res)
    kept = apply_group_thresholds(fc, taus, metric)
    official = score(kept, gt, metric)["final_score"]
    print(f"{name}: по кривым {est:.6f}; официальный {official:.6f}; оставлено {len(kept['features'])}",
          flush=True)
    return official, taus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--las49", default="")
    ap.add_argument("--las-new", default="")
    ap.add_argument("--centers", default="", help="круги карты центров вне обучения; своя группа порога sp_focal")
    args = ap.parse_args()
    work = Path(args.work)
    sys.path.insert(0, str(work))
    from score_coarse import las_boxes, tag

    boxes = las_boxes()
    gt = load(work / "gt_train.geojson")
    fused = load(work / "fusion" / "li_hard2_fused.geojson")
    print(f"помечено li_coarse: {tag(fused, boxes)}", flush=True)
    gor = load(work / "spor_gor_oof.geojson")
    geo = load(work / "geo_new.geojson")
    feats = fused["features"] + gor["features"] + geo["features"]
    if args.centers:
        extra = load(Path(args.centers))
        keep = {"kurgany_povrezhdennye", "kurgany_tselye"}
        n = 0
        for f in extra["features"]:
            p = dict(f.get("properties") or {})
            if p.get("class_name") not in keep:
                continue
            p["threshold_group"] = "sp_focal"
            p["markup_type"] = p.get("markup_type") or "sp_or"
            feats.append({**f, "properties": p})
            n += 1
        print(f"кругов карты центров {n}", flush=True)
    fc = {"type": "FeatureCollection", "features": feats}
    base, taus = report("полный стек", fc, gt, args.metric)
    print(f"полный стек {base:.6f}", flush=True)
    from expds.metric import apply_group_thresholds, score
    for label, path in (("49 с LAS", args.las49), ("34 новых", args.las_new)):
        if not path:
            continue
        regs = names(Path(path))
        held = keep_regions(fc, regs)
        held_gt = keep_regions(gt, regs)
        kept = apply_group_thresholds(held, taus, args.metric)
        moved = score(kept, held_gt, args.metric)["final_score"]
        print(f"{label}: порог со всех участков {moved:.6f}", flush=True)
        report(f"{label}, порог на этом же подмножестве", held, held_gt, args.metric)
    print("фортификации аэрофото в этом счёте нет: их предсказаний вне обучения на диске нет", flush=True)


if __name__ == "__main__":
    main()
