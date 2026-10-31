"""Отдельный порог для лидара, у которого сетка DEM крупнее 0,1 м.

Предсказания OOF не хранят имя LAS. Облако определяется по попаданию точки
полигона в границы заголовка LAS; шаг сетки — тот же расчёт, что у инференса.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.accel import grid_spec  # noqa: E402
from expds.catalog import scan_root  # noqa: E402
from expds.crs import WEB_MERCATOR, transformer  # noqa: E402
from expds.metric import (  # noqa: E402
    apply_group_thresholds,
    build_curves,
    official_weighted,
    score,
    sweep_groups,
)

W = Path(r"F:\expds3\work")
METRIC = r"C:\Sloi\files\materials\compute_metrics.py"
ROOT = r"F:\expds3\train-expds-3"


def first_xy(geom: dict) -> tuple[float, float]:
    c = geom["coordinates"]
    while c and isinstance(c[0], list):
        c = c[0]
    return float(c[0]), float(c[1])


def las_boxes() -> dict[str, list[tuple]]:
    boxes: dict[str, list[tuple]] = {}
    for reg in scan_root(ROOT):
        if not reg.las:
            continue
        rows = []
        for las in reg.las:
            with laspy.open(las) as f:
                h = f.header
            spec = grid_spec(float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]),
                             0.1, 60_000_000)
            rows.append((float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]),
                         int(spec["factor"]), las.name))
        boxes[reg.name] = rows
    n_coarse = sum(f > 1 for rows in boxes.values() for *_, f, _n in rows)
    print(f"облаков с шагом > 0,1 м: {n_coarse} из {sum(len(v) for v in boxes.values())}", flush=True)
    return boxes


def tag(fc: dict, boxes: dict[str, list[tuple]]) -> int:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, f in enumerate(fc["features"]):
        p = f["properties"]
        if p.get("markup_type") != "li" or p.get("threshold_group"):
            continue
        groups[(p.get("region_name") or "", p.get("original_crs") or "")].append(i)
    n = 0
    for (region, crs), idxs in groups.items():
        rows = boxes.get(region) or []
        if not rows:
            continue
        if len(rows) == 1:
            factors = np.full(len(idxs), rows[0][4])
        else:
            xs, ys = zip(*(first_xy(fc["features"][i]["geometry"]) for i in idxs))
            if crs and crs != WEB_MERCATOR:
                xs, ys = transformer(WEB_MERCATOR, crs).transform(np.array(xs), np.array(ys))
                xs, ys = np.asarray(xs), np.asarray(ys)
            else:
                xs, ys = np.asarray(xs), np.asarray(ys)
            factors = np.ones(len(idxs), np.int32)
            claimed = np.zeros(len(idxs), bool)
            for minx, miny, maxx, maxy, factor, _name in rows:
                inside = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy) & ~claimed
                factors[inside] = factor
                claimed |= inside
        for i, factor in zip(idxs, factors):
            if int(factor) > 1:
                fc["features"][i]["properties"]["threshold_group"] = "li_coarse"
                n += 1
    return n


def report(name: str, fc: dict, gt: dict, regions: set[str] | None) -> None:
    def keep(feat_fc):
        if regions is None:
            return feat_fc
        feats = [f for f in feat_fc["features"]
                 if (f.get("properties") or {}).get("region_name") in regions]
        return {"type": "FeatureCollection", "features": feats}

    pred, gtf = keep(fc), keep(gt)
    curves = build_curves(pred, gtf, METRIC)
    res = sweep_groups(curves)
    taus: dict[str, dict[str, float]] = {}
    for cls, v in res.items():
        if v["f1"] <= 0 and not v["groups"]:
            continue
        print(f"{name} {cls:24s} GT {v['n_gt']:5d}  F1 {v['f1']:.4f}", flush=True)
        for g, gv in v["groups"].items():
            taus.setdefault(g, {})[cls] = gv["threshold"]
            if gv["kept"]:
                print(f"    {g:10s} порог {gv['threshold']:.6f}  оставлено {gv['kept']:6d}  TP {gv['tp']:5d}",
                      flush=True)
    official = score(apply_group_thresholds(pred, taus, METRIC), gtf, METRIC)["final_score"]
    print(f"{name} балл по кривым {official_weighted(res):.6f}; официальный {official:.6f}", flush=True)


def main() -> None:
    boxes = las_boxes()
    print("читаю предсказания", flush=True)
    fc = json.loads((W / "fusion" / "li_hard2_fused.geojson").read_text(encoding="utf-8"))
    gt = json.loads((W / "gt_train.geojson").read_text(encoding="utf-8"))
    n = tag(fc, boxes)
    print(f"помечено li_coarse: {n}", flush=True)
    report("все участки", fc, gt, None)


if __name__ == "__main__":
    main()
