"""Общий балл: лидар с раздельным порогом сетки и городища спутника
против того же плюс курганы сети 1 м, у которых уверенность заменена
рангом внутри участка.

Класс в метрике общий для всех модальностей, поэтому слой сам по себе
не говорит, вырастет ли балл. Точная цифра — официальный расчёт после порога.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, r"F:\expds3\work")

from expds.metric import (  # noqa: E402
    apply_group_thresholds,
    build_curves,
    official_weighted,
    score,
    sweep_groups,
)
from rescore_rank import percentile, groups  # noqa: E402
from score_coarse import tag, las_boxes  # noqa: E402

W = Path(r"F:\expds3\work")
METRIC = r"C:\Sloi\files\materials\compute_metrics.py"
KEEP = {"kurgany_povrezhdennye", "kurgany_tselye"}


def load(path: Path) -> dict:
    print(f"читаю {path.name}", flush=True)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def rank_satellite(fc: dict) -> list[dict]:
    feats = [f for f in fc["features"]
             if (f.get("properties") or {}).get("class_name") in KEEP
             and (f.get("properties") or {}).get("markup_type") in (None, "sp_or")]
    raw = np.array([float((f.get("properties") or {}).get("confidence") or 0.0) for f in feats])
    keys = [(f.get("properties") or {}).get("region_name") or "" for f in feats]
    cls = [(f.get("properties") or {}).get("class_name") or "" for f in feats]
    pct = percentile(raw, groups([a + "\0" + b for a, b in zip(keys, cls)]))
    out = []
    for f, c in zip(feats, pct):
        props = dict(f["properties"])
        props["confidence"] = round(float(c), 6)
        props["threshold_group"] = "sp_r1"
        props["markup_type"] = "sp_or"
        out.append({**f, "properties": props})
    print(f"курганов сети 1 м: {len(out)}", flush=True)
    return out


def report(name: str, feats: list, gt: dict) -> float:
    fc = {"type": "FeatureCollection", "features": feats}
    print(f"{name}: объектов {len(feats)}", flush=True)
    curves = build_curves(fc, gt, METRIC)
    res = sweep_groups(curves)
    taus: dict[str, dict[str, float]] = {}
    for cls, v in res.items():
        if v["f1"] <= 0 and not any(g["kept"] for g in v["groups"].values()):
            continue
        print(f"{name} {cls:24s} GT {v['n_gt']:5d}  F1 {v['f1']:.4f}", flush=True)
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
            if gv["kept"]:
                print(f"    {g:10s} порог {gv['threshold']:.4f}  оставлено {gv['kept']:6d}  TP {gv['tp']:5d}",
                      flush=True)
    est = official_weighted(res)
    kept = apply_group_thresholds(fc, taus, METRIC)
    print(f"{name} оставлено {len(kept['features'])}; балл по кривым {est:.6f}", flush=True)
    official = score(kept, gt, METRIC)["final_score"]
    print(f"{name} официальный {official:.6f}", flush=True)
    del curves, fc, kept
    gc.collect()
    return official


def main() -> None:
    boxes = las_boxes()
    gt = load(W / "gt_train.geojson")
    fused = load(W / "fusion" / "li_hard2_fused.geojson")
    n = tag(fused, boxes)
    print(f"помечено li_coarse: {n}", flush=True)
    gor = load(W / "spor_gor_oof.geojson")
    base = fused["features"] + gor["features"]
    del fused, gor
    gc.collect()
    base_score = report("без сети 1 м", base, gt)
    r1 = rank_satellite(load(W / "spor_r1_oof.geojson"))
    plus_score = report("с рангом сети 1 м", base + r1, gt)
    print(f"разница {plus_score - base_score:+.6f}", flush=True)


if __name__ == "__main__":
    main()
