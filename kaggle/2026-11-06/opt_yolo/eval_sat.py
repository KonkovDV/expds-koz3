#!/usr/bin/env python3
"""Спутник на участках фолда по официальной метрике (materials/compute_metrics.py): YOLO OOF против стека.

    python eval_sat.py --fold 4 --pred NAME=PRED.geojson[:GROUPS] [--pred ...] [--thr 0.1,0.2,...] [--out OUT.json]

- Эталон — gt_train_v2.geojson, только markup_type sp_or, только участки фолда (folds_family_k5.json),
  у которых есть окна в выгрузке (index.csv). Предсказания — те же участки; :GROUPS — фильтр threshold_group
  через «+» (например stack.geojson:sp_or+sp_focal+sp_focal2+sp_arch).
- Для каждого порога из --thr (один на все классы) — P/R/F1 спутника и F1 по классам. Порог выбран на тех же
  данных: это ориентир «потолка», а не честная оценка; честные пороги — по другим фолдам (следующий шаг).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GT = Path("/workspace/koz3/w5data/gt_train_v2.geojson")
FOLDS = REPO / "kaggle/2026-10-28/nolidar_verifier/folds_family_k5.json"


def calculator():
    spec = importlib.util.spec_from_file_location("cm", REPO / "materials" / "compute_metrics.py")
    cm = importlib.util.module_from_spec(spec)
    sys.modules["cm"] = cm
    spec.loader.exec_module(cm)
    return cm.F1Calculator()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--pred", action="append", required=True)
    ap.add_argument("--thr", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    ap.add_argument("--index", type=Path, default=Path("/workspace/yolo_spor/index.csv"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    fold_sites = {r for f in folds["folds"] if f["fold"] == a.fold for r in f["test_regions"]}
    have = {r["site"] for r in csv.DictReader(open(a.index, encoding="utf-8"))}
    sites = fold_sites & have
    gt_all = json.loads(GT.read_text(encoding="utf-8"))
    gt = {"type": "FeatureCollection", "features": [f for f in gt_all["features"]
                                                    if f["properties"].get("markup_type") == "sp_or"
                                                    and f["properties"].get("region_name") in sites]}
    calc = calculator()
    res = {"fold": a.fold, "sites": len(sites), "gt": len(gt["features"]), "runs": {}}
    for spec_ in a.pred:
        name, path = spec_.split("=", 1)
        groups = None
        if ":" in path:
            path, g = path.rsplit(":", 1)
            groups = set(g.split("+"))
        fs = [f for f in json.loads(Path(path).read_text(encoding="utf-8"))["features"]
              if f["properties"].get("region_name") in sites
              and (groups is None or f["properties"].get("threshold_group") in groups)]
        for t in [float(x) for x in a.thr.split(",")]:
            sel = [f for f in fs if float(f["properties"].get("confidence", 0)) >= t]
            r = calc.calculate_f1({"type": "FeatureCollection", "features": sel}, gt)
            dt = {k: v for k, v in (r.get("by_data_type") or {}).items() if "sp" in k.lower() or "спут" in k.lower()}
            pc = {c: round(v.get("f1", 0.0), 4) for c, v in (r.get("per_class") or {}).items() if isinstance(v, dict)}
            res["runs"].setdefault(name, []).append({"thr": t, "n": len(sel), "final": round(r["final_score"], 5),
                                                     "sat": dt, "f1_by_class": pc})
            print(name, f"thr {t:.2f}", "n", len(sel), "final", round(r["final_score"], 5), json.dumps(dt, ensure_ascii=False)[:200],
                  flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
