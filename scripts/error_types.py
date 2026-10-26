#!/usr/bin/env python3
"""Типы ошибок по официальному сопоставлению: где именно теряется балл класса.

Сопоставление — то же жадное, что у `compute_metrics.py` (через `expds.metric`):
по убыванию уверенности, один к одному, попадание по правилу метрики. После него
каждый ложный объект и каждый пропуск получает один тип.

Ложные (FP), по приоритету:
  localization    — лежит на эталоне своего класса, который остался пропущенным:
                    объект на месте, но не прошёл правило (покрытие < 0,5, внутренняя
                    точка вне эталона) или эталон забрал другой объект;
  duplicate       — лежит на эталоне своего класса, который уже найден;
  class_confusion — прошёл бы правило на эталоне другого класса;
  background      — всё остальное.
Пропуски (FN), по приоритету:
  covered         — на нём лежит объект своего класса;
  other_class     — объект другого класса прошёл бы правило на нём;
  missed          — рядом нет ничего.

«Лежит на» — пересечение с полигоном эталона или центр ближе 2·r_tolerance к точке.
Для каждого типа считается, сколько балла вернёт его исправление (`gain`): F1 класса
при TP + k, FP − k, FN − k (место), FP − k (дубли, фон), TP + k, FN − k (класс),
в весах метрики. Это верхняя граница, а не обещание.

Вход — уже отфильтрованный порогами набор, например `scoreboard.py --dump-kept`.
Выход с именами участков — только в рабочую папку, не в notes/.

    python scripts/scoreboard.py ... --dump-kept F:/expds3/work/kept_base.geojson
    python scripts/error_types.py --pred F:/expds3/work/kept_base.geojson --gt F:/expds3/work/gt_train_v2.geojson \
        --out F:/expds3/work/errors_base.json --geojson F:/expds3/work/errors_base.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds.metric import load_official

FP_TYPES = ("localization", "duplicate", "class_confusion", "background")
FN_TYPES = ("covered", "other_class", "missed")
SIZE_BINS = (100.0, 500.0, 1000.0, 5000.0, 10000.0)


def _collect(mod, calc, fc: dict, is_gt: bool) -> dict[str, list[dict]]:
    """Как F1Calculator._collect_by_class, но с исходным объектом, регионом и слоем."""
    out: dict[str, list[dict]] = {}
    for idx, feature in enumerate(fc.get("features", [])):
        props = feature.get("properties", {}) or {}
        cls = mod.normalize_class_name(props.get("class_name", "unknown"))
        if cls is None or cls not in calc.target_classes:
            continue
        geom = mod._extract_geometry(feature)
        if geom is None:
            continue
        geom = mod._to_utm(geom, calc.input_crs_mode)
        if geom.geom_type == "Point" and not is_gt:
            geom = mod._point_to_circle(geom, calc.r_tolerance, calc.circle_points)
        out.setdefault(cls, []).append({
            "geom": geom, "conf": mod._get_confidence(feature), "idx": idx,
            "region": str(props.get("region_name", "default_source")),
            "markup": str(props.get("markup_type") or ""),
        })
    return out


def _size_bin(geom) -> str:
    if geom.geom_type == "Point":
        return "point"
    area = geom.area
    for edge in SIZE_BINS:
        if area <= edge:
            return f"<={int(edge)}"
    return f">{int(SIZE_BINS[-1])}"


def _f1(tp: float, fp: float, fn: float, eps: float = 1e-6) -> float:
    """F1 класса ровно как в F1Calculator.calculate_f1 (с его эпсилоном)."""
    precision = tp / (tp + 0.5 * fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def classify(pred: dict, gt: dict, metric_path: str | None = None) -> dict:
    """Сопоставление по правилу метрики и тип каждой ошибки. Возвращает сводку и сами ошибки."""
    from shapely.strtree import STRtree

    mod = load_official(metric_path)
    calc = mod.F1Calculator()
    preds, gts = _collect(mod, calc, pred, False), _collect(mod, calc, gt, True)
    near = 2.0 * calc.r_tolerance

    def footprint(g):
        return g.buffer(near) if g.geom_type == "Point" else g

    trees = {c: STRtree([footprint(x["geom"]) for x in v]) for c, v in gts.items() if v}
    matched_gt: dict[str, set[int]] = {}
    hit_pred: dict[str, set[int]] = {}
    for cls in calc.target_classes:
        p, g = preds.get(cls, []), gts.get(cls, [])
        order = sorted(range(len(p)), key=lambda i: p[i]["conf"], reverse=True)
        taken: set[int] = set()
        hits: set[int] = set()
        if p and g:
            gg = [x["geom"] for x in g]
            tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == "Point" else x for x in gg])
            for i in order:
                pg = p[i]["geom"]
                if pg.geom_type != "Polygon":
                    continue
                best, best_j = float("-inf"), -1
                for j in sorted(int(k) for k in tree.query(pg)):
                    if j in taken:
                        continue
                    ok, s = calc._match_quality(pg, gg[j])
                    if ok and s > best + 1e-9:
                        best, best_j = s, j
                if best_j >= 0:
                    taken.add(best_j)
                    hits.add(i)
        matched_gt[cls], hit_pred[cls] = taken, hits

    def lies_on(pg, gg) -> bool:
        if gg.geom_type == "Point":
            return pg.centroid.distance(gg) <= near
        return pg.intersects(gg)

    pred_trees = {c: STRtree([y["geom"] for y in v]) for c, v in preds.items() if v}
    errors: list[dict] = []
    per: dict[str, dict] = {}
    for cls in calc.target_classes:
        p, g = preds.get(cls, []), gts.get(cls, [])
        fp_types, fn_types = Counter(), Counter()
        fp_markup, fn_markup, fn_size = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        for i, x in enumerate(p):
            if i in hit_pred[cls]:
                continue
            pg, kind = x["geom"], "background"
            same = [int(j) for j in trees[cls].query(pg)] if cls in trees else []
            on = [j for j in same if lies_on(pg, g[j]["geom"])]
            if any(j not in matched_gt[cls] for j in on):
                kind = "localization"
            elif on:
                kind = "duplicate"
            else:
                for other, tree in trees.items():
                    if other == cls:
                        continue
                    if any(calc._match_quality(pg, gts[other][int(j)]["geom"])[0] for j in tree.query(pg)):
                        kind = "class_confusion"
                        break
            fp_types[kind] += 1
            fp_markup[kind][x["markup"]] += 1
            errors.append({"kind": "FP", "type": kind, "class_name": cls, "region": x["region"],
                           "markup": x["markup"], "conf": x["conf"], "idx": x["idx"], "side": "pred"})
        for j, y in enumerate(g):
            if j in matched_gt[cls]:
                continue
            gg, kind = y["geom"], "missed"
            probe = footprint(gg)
            if cls in pred_trees and any(lies_on(p[int(i)]["geom"], gg) for i in pred_trees[cls].query(probe)):
                kind = "covered"
            else:
                for other, tree in pred_trees.items():
                    if other == cls:
                        continue
                    if any(calc._match_quality(preds[other][int(i)]["geom"], gg)[0] for i in tree.query(probe)):
                        kind = "other_class"
                        break
            fn_types[kind] += 1
            fn_markup[kind][y["markup"]] += 1
            fn_size[kind][_size_bin(gg)] += 1
            errors.append({"kind": "FN", "type": kind, "class_name": cls, "region": y["region"],
                           "markup": y["markup"], "idx": y["idx"], "side": "gt"})
        tp = len(hit_pred[cls])
        fp, fn = len(p) - tp, len(g) - tp
        per[cls] = {"n_gt": len(g), "tp": tp, "fp": fp, "fn": fn, "f1": _f1(tp, fp, fn, calc.epsilon),
                    "fp_types": dict(fp_types), "fn_types": dict(fn_types),
                    "fp_by_markup": {k: dict(v) for k, v in fp_markup.items()},
                    "fn_by_markup": {k: dict(v) for k, v in fn_markup.items()},
                    "fn_by_size": {k: dict(v) for k, v in fn_size.items()}}

    weights = mod.CLASS_WEIGHTS
    present = [c for c, v in per.items() if v["n_gt"] > 0]
    total_w = sum(weights.get(c, 0.0) for c in present) or 1.0
    score = sum(per[c]["f1"] * weights.get(c, 0.0) for c in present) / total_w
    for c in present:
        v = per[c]
        tp, fp, fn, f1 = v["tp"], v["fp"], v["fn"], v["f1"]
        w = weights.get(c, 0.0) / total_w
        k_loc = min(v["fn_types"].get("covered", 0), v["fp_types"].get("localization", 0))
        k_cls = v["fn_types"].get("other_class", 0)
        v["loss"] = round(w * (1 - f1), 6)
        e = calc.epsilon
        v["gain"] = {
            "localization": round(w * (_f1(tp + k_loc, fp - k_loc, fn - k_loc, e) - f1), 6),
            "duplicate": round(w * (_f1(tp, fp - v["fp_types"].get("duplicate", 0), fn, e) - f1), 6),
            "background": round(w * (_f1(tp, fp - v["fp_types"].get("background", 0), fn, e) - f1), 6),
            "class_confusion": round(w * (_f1(tp + k_cls, fp, fn - k_cls, e) - f1), 6),
        }
    return {"score": score, "per_class": per, "errors": errors}


def errors_geojson(res: dict, pred: dict, gt: dict) -> dict:
    """Ошибки исходными геометриями (EPSG:3857) для QGIS."""
    feats = []
    for e in res["errors"]:
        src = (pred if e["side"] == "pred" else gt)["features"][e["idx"]]
        props = {k: v for k, v in e.items() if k not in ("idx", "side")}
        feats.append({"type": "Feature", "geometry": src.get("geometry"), "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="набор после порогов (scoreboard.py --dump-kept)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--out", required=True, help="сводка JSON (в рабочую папку: внутри имена участков)")
    ap.add_argument("--geojson", default=None, help="ошибки геометриями для QGIS")
    args = ap.parse_args()
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    res = classify(pred, gt, args.metric)
    by_region: dict[str, Counter] = defaultdict(Counter)
    for e in res["errors"]:
        by_region[e["region"]][f"{e['kind']}:{e['class_name']}:{e['type']}"] += 1
    summary = {"score": res["score"], "per_class": res["per_class"],
               "by_region": {r: dict(c) for r, c in sorted(by_region.items())}}
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.geojson:
        Path(args.geojson).write_text(json.dumps(errors_geojson(res, pred, gt), ensure_ascii=False), encoding="utf-8")
    print(f"балл {res['score']:.6f}")
    for cls, v in sorted(res["per_class"].items(), key=lambda kv: -kv[1].get("loss", 0)):
        if not v["n_gt"] and not v["fp"]:
            continue
        gain = v.get("gain", {})
        print(f"{cls:22s} F1 {v['f1']:.3f} потеря {v.get('loss', 0):.4f} | FP {v['fp_types']} | FN {v['fn_types']}")
        if gain:
            print(" " * 23 + "вернёт: " + ", ".join(f"{k} +{g:.4f}" for k, g in sorted(gain.items(), key=lambda t: -t[1])))


if __name__ == "__main__":
    main()
