"""Общий балл: лидар + городища спутника против того же плюс курганы/архитектура спутника.

Порог каждой пары (слой, класс) подбирается на общий F1 класса. Эталон — все участки.
"""
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")

from expds.metric import (
    apply_group_thresholds,
    build_curves,
    official_weighted,
    score,
    sweep_groups,
)

W = Path(r"F:\expds3\work")
METRIC = r"C:\Sloi\files\materials\compute_metrics.py"


def load(path: Path) -> dict:
    print(f"читаю {path.name}", flush=True)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def report(name: str, feats: list, gt: dict) -> float:
    fc = {"type": "FeatureCollection", "features": feats}
    print(f"{name}: объектов {len(feats)}", flush=True)
    curves = build_curves(fc, gt, METRIC)
    res = sweep_groups(curves)
    taus: dict[str, dict[str, float]] = {}
    for cls, v in res.items():
        print(f"{name} {cls:24s} GT {v['n_gt']:5d}  F1 {v['f1']:.4f}", flush=True)
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
            if gv["kept"]:
                print(f"    {g:10s} порог {gv['threshold']:.4f}  оставлено {gv['kept']:6d}  TP {gv['tp']:5d}",
                      flush=True)
    est = official_weighted(res)
    official = score(apply_group_thresholds(fc, taus, METRIC), gt, METRIC)["final_score"]
    print(f"{name} балл по кривым {est:.6f}; официальный {official:.6f}", flush=True)
    del curves, fc
    gc.collect()
    return official


def main() -> None:
    gt = load(W / "gt_train.geojson")
    li = load(W / "fusion" / "li_hard2_fused.geojson")
    gor = load(W / "spor_gor_oof.geojson")
    base = li["features"] + gor["features"]
    del li, gor
    gc.collect()
    base_score = report("без курганов спутника", base, gt)
    sp = load(W / "spor_all_oof.geojson")
    plus = base + sp["features"]
    del sp
    gc.collect()
    plus_score = report("со спутником", plus, gt)
    print(f"разница {plus_score - base_score:+.6f}", flush=True)


if __name__ == "__main__":
    main()
