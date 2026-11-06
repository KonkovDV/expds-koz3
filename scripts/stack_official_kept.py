#!/usr/bin/env python3
"""Официальный compute_metrics.py на ВСЁМ оставленном стеке одним вызовом (без раскладки на кластеры) —.

Пороги групп берутся из --out stack_scoreboard.py (sweep по всему стеку). Оставленные объекты
(apply_group_thresholds — пообъектный фильтр) собираются потоково из --split-dir в порядке файла,
затем: F1Calculator.calculate_f1(kept, gt) целиком (глобальное сопоставление, как на платформе) и
быстрый build_curves(kept, gt) (scope по умолчанию = global) → score_at. Сравниваются итог и TP/FP/FN.

python scripts/stack_official_kept.py --proof OUT_FAST.json --gt GT --split-dir DIR --out RES.json [--kept-out K.geojson]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from expds import metric as M  # noqa: E402
from stack_scoreboard import read_family  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proof", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kept-out", type=Path)
    a = ap.parse_args()
    P = json.loads(a.proof.read_text(encoding="utf-8"))
    taus = P["taus"]
    mod = M.load_official(a.metric)
    calc = mod.F1Calculator()
    rows = []
    fams = sorted(json.loads((a.split_dir / "_counts.json").read_text(encoding="utf-8"))["families"])
    for fam in fams:
        fr = read_family(a.split_dir, fam)
        kept = M.apply_group_thresholds({"type": "FeatureCollection", "features": [f for _, f in fr]}, taus, a.metric)["features"]
        ids = {id(f): i for i, f in fr}
        for f in kept:
            rows.append((ids.get(id(f)), f))
        if any(i is None for i, _ in rows):   # apply_group_thresholds вернул копии — индекс по содержимому
            raise SystemExit("apply_group_thresholds копирует объекты: нужен другой способ сохранить порядок")
    rows.sort(key=lambda r: r[0])
    kept = {"type": "FeatureCollection", "features": [f for _, f in rows]}
    if a.kept_out:
        a.kept_out.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    gt = json.loads(a.gt.read_bytes())
    t = time.perf_counter()
    fast_score, per = M.score_at(M.build_curves(kept, gt, a.metric), {})
    t_fast = time.perf_counter() - t
    t = time.perf_counter()
    off = calc.calculate_f1(kept, gt)
    t_off = time.perf_counter() - t
    cls = {}
    for c in calc.target_classes:
        o = off["per_class"][c]
        cls[c] = {"official": [o["tp"], o["fp"], o["fn"]], "fast": [per[c]["tp"], per[c]["fp"], per[c]["fn"]],
                  "equal": (o["tp"], o["fp"], o["fn"]) == (per[c]["tp"], per[c]["fp"], per[c]["fn"]),
                  "f1_official": o.get("f1"), "f1_fast": per[c]["f1"]}
    res = {"n_kept": len(kept["features"]), "n_gt": len(gt["features"]),
           "official_final_score": off["final_score"], "fast_final_score": fast_score,
           "cluster_run_base_all": P["base"][0],
           "delta_fast_vs_official": fast_score - off["final_score"],
           "delta_cluster_vs_official": P["base"][0] - off["final_score"],
           "per_class": cls, "all_counts_equal": all(v["equal"] for v in cls.values()),
           "t_fast_s": round(t_fast, 2), "t_official_s": round(t_off, 2),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, ensure_ascii=False))
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
