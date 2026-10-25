#!/usr/bin/env python3
"""Сдвиг разметки между слоями по участкам: есть ли у снимка устойчивый сдвиг относительно лидара.

Если эталон аэрофото или спутника нарисован по снимку, привязанному со сдвигом, копия
кургана лидара в этот слой промахивается мимо эталона (внутренняя точка не внутри), а
сеть снимка учит смещённую точку. Скрипт считает по каждому участку и паре слоёв:

- перенос dst относительно ref (мода смещений ближе --r0, уточнение медианой ближе --r1);
- устойчив ли он (не меньше 5 пар, больше 1,5 м и трёх стандартных ошибок);
- долю центров ref внутри полигонов dst до переноса и после — прямая мера того, сколько
  копий ref → dst попало бы в эталон.

С --pred то же для предсказаний вне обучения против эталона своего слоя: систематический
сдвиг сети относительно разметки.

    python scripts/layer_shift.py --gt F:/expds3/work/gt_train_v2.geojson --out notes/layer_shift_gt.json
    python scripts/layer_shift.py --gt F:/expds3/work/gt_train_v2.geojson \
        --pred F:/expds3/work/spor_focal_oof.geojson --pred-markups sp_or --pred-min 0.36 \
        --out notes/layer_shift_spor_pred.json

Имена участков трейна в выводе есть (как во всех notes/); выборку и предсказания в git не класть.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")


def load_layers(path: str, classes: tuple[str, ...], markups: set[str] | None = None,
                min_conf: float | None = None) -> dict[str, dict[str, list]]:
    """region → markup → [геометрии EPSG:3857] выбранных классов."""
    from shapely.geometry import shape

    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in fc.get("features", []):
        p = f.get("properties") or {}
        if p.get("class_name") not in classes:
            continue
        mt = p.get("markup_type") or ""
        if markups is not None and mt not in markups:
            continue
        if min_conf is not None and float(p.get("confidence", 1.0)) < min_conf:
            continue
        try:
            g = shape(f["geometry"])
        except Exception:
            continue
        if not g.is_empty:
            out[p.get("region_name") or ""][mt].append(g)
    return out


def to_utm(geoms: list, utm: str) -> tuple[np.ndarray, list]:
    """Геометрии EPSG:3857 → (внутренние точки N×2, полигоны) в метрах зоны."""
    from expds.crs import WEB_MERCATOR, to_crs

    polys = [to_crs(g, WEB_MERCATOR, utm) for g in geoms]
    pts = np.array([[p.representative_point().x, p.representative_point().y] for p in polys]).reshape(-1, 2)
    return pts, polys


def compare(ref_geoms: list, dst_geoms: list, r0: float, r1: float) -> dict | None:
    from expds import register

    if not ref_geoms or not dst_geoms:
        return None
    allp = [g.representative_point() for g in ref_geoms + dst_geoms]
    utm = register.utm_of(np.array([p.x for p in allp]), np.array([p.y for p in allp]))
    ref, _ = to_utm(ref_geoms, utm)
    dst, dst_polys = to_utm(dst_geoms, utm)
    s = register.estimate_shift(ref, dst, r0=r0, r1=r1)
    row = {"n_ref": len(ref), "n_dst": len(dst), **s.as_dict(),
           "inside_before": round(register.contain_rate(ref, dst_polys), 4)}
    row["inside_after"] = (round(register.contain_rate(ref, dst_polys, (s.dx, s.dy)), 4)
                           if s.n_pairs else row["inside_before"])
    return row


def summarize(rows: list[dict]) -> dict:
    sysrows = [r for r in rows if r["systematic"]]
    n_ref = sum(r["n_ref"] for r in rows)
    before = sum(r["inside_before"] * r["n_ref"] for r in rows)
    after = sum((r["inside_after"] if r["systematic"] else r["inside_before"]) * r["n_ref"] for r in rows)
    return {"regions": len(rows), "systematic": len(sysrows),
            "median_shift_systematic_m": round(float(np.median([r["shift_m"] for r in sysrows])), 2) if sysrows else 0.0,
            "ref_points": n_ref, "inside_before": round(before), "inside_after_systematic_only": round(after)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pairs", nargs="*", default=["li:ae", "li:sp_or", "li:or", "ae:sp_or"],
                    help="ref:dst — сдвиг слоя dst относительно ref")
    ap.add_argument("--classes", nargs="*", default=list(KURGANS))
    ap.add_argument("--pred", default="", help="предсказания вне обучения (вместо пар слоёв эталона)")
    ap.add_argument("--pred-markups", nargs="*", default=["sp_or", "ae"])
    ap.add_argument("--pred-min", type=float, default=None, help="предсказания слабее не берутся")
    ap.add_argument("--r0", type=float, default=30.0, help="радиус поиска моды смещений, м")
    ap.add_argument("--r1", type=float, default=6.0, help="радиус пар при уточнении, м")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    classes = tuple(args.classes)
    gt = load_layers(args.gt, classes)
    report: dict[str, dict] = {}
    if args.pred:
        pred = load_layers(args.pred, classes, set(args.pred_markups), args.pred_min)
        pairs = [(f"pred_{m}", m) for m in args.pred_markups]
        for name, m in pairs:
            rows = []
            for region, layers in sorted(gt.items()):
                row = compare(pred.get(region, {}).get(m, []), layers.get(m, []), args.r0, args.r1)
                if row:
                    rows.append({"region": region, **row})
            report[f"{name}:{m}"] = {"summary": summarize(rows), "regions": rows}
    else:
        for pair in args.pairs:
            a, b = pair.split(":")
            rows = []
            for region, layers in sorted(gt.items()):
                row = compare(layers.get(a, []), layers.get(b, []), args.r0, args.r1)
                if row:
                    rows.append({"region": region, **row})
            report[pair] = {"summary": summarize(rows), "regions": rows}
    for key, rep in report.items():
        s = rep["summary"]
        print(f"{key}: участков {s['regions']}, устойчивый сдвиг у {s['systematic']} "
              f"(медиана {s['median_shift_systematic_m']} м); центров ref {s['ref_points']}, "
              f"внутри эталона dst {s['inside_before']} → {s['inside_after_systematic_only']} после переноса",
              flush=True)
        for r in rep["regions"]:
            if r["systematic"]:
                print(f"    {r['region']}: ({r['dx']}, {r['dy']}) м, пар {r['n_pairs']}, MAD {r['mad_m']} м, "
                      f"внутри {r['inside_before']} → {r['inside_after']}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"args": vars(args), "report": report}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
