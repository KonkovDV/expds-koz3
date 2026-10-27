#!/usr/bin/env python3
"""Потолок геометрии выпуска: идеальный детектор, форма как у сдачи, официальная метрика.

Каждый объект эталона выдаётся с уверенностью 1. Форма берётся из сдачи, не из контура
разметки, кроме классов, которые сдача и так пишет полигоном:

- курганы лидара, мозаики, канала σ и копий в аэрофото — круг 2 м (`candidates.circle_m`);
- курганы спутника — круг 4 м (карта центров);
- архитектура — круг 2 м;
- находки — круг 0,3 м;
- городища, фортификации и object_poly — полигон эталона: сдача пишет маску.

Второй прогон ставит вместо маски городища круг 3 м (`hillfort.core_m`), вместо
фортификации — круг 2 м. Полнота полигона ниже 0,9 — дыра в `finalize` или склейке.
Полнота полигона высокая, а круга низкая — круг этот класс не держит, маска держит.
Оба высоких — пустая корзина площадей на площадке сидит в детекторе, не в выпуске.

Круги одного слоя не склеиваются между собой: источник один на участок и слой,
как у одного файла. Дальше `watchdog.finish`: порог, склейка разных снимков,
`finalize`, лимит файла. Имена участков в сводку не попадают.

    python scripts/oracle_ceiling.py --gt F:/expds3/work/gt_train_v2.geojson \
        --metric C:/Sloi/files/materials/compute_metrics.py \
        --out F:/expds3/work/oracle_ceiling.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from shapely.geometry import shape
from shapely.validation import make_valid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Радиус круга сдачи, метры на земле. None — в выпуск идёт полигон эталона.
_KURGAN = ("kurgany_povrezhdennye", "kurgany_tselye")


def radius_m(cls: str, markup: str, masks_as_circles: bool) -> float | None:
    """Радиус круга или None, если объект остаётся полигоном эталона."""
    if cls in _KURGAN:
        return 4.0 if markup == "sp_or" else 2.0
    if cls == "arkhitektury":
        return 2.0
    if cls == "finds_points":
        return 0.3
    if masks_as_circles and cls == "gorodishcha":
        return 3.0
    if masks_as_circles and cls == "fortifikatsii":
        return 2.0
    return None


def _polygon(geom):
    """Крупнейший полигон. Точка и пустая геометрия остаются как есть."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == "Polygon" or geom.geom_type == "Point":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if not g.is_empty and g.area > 0]
    return max(parts, key=lambda g: g.area) if parts else geom


def emit(gt: dict, masks_as_circles: bool = False) -> list[dict]:
    """Объекты эталона в форме сдачи. Уверенность 1, один источник на участок и слой."""
    from expds.raster import circle_3857

    out = []
    for feat in gt.get("features", []):
        props = feat.get("properties") or {}
        cls = props.get("class_name")
        if not cls:
            continue
        geom = _polygon(shape(feat["geometry"]))
        if geom is None:
            continue
        markup = str(props.get("markup_type") or "")
        region = str(props.get("region_name") or "default_source")
        radius = radius_m(cls, markup, masks_as_circles)
        if radius is not None:
            pt = geom if geom.geom_type == "Point" else geom.representative_point()
            geom = circle_3857(pt.x, pt.y, radius, "EPSG:3857")
        out.append({"geom": geom, "class_name": cls, "confidence": 1.0, "region_name": region,
                    "markup_type": markup, "source": f"{region}|{markup}", "original_crs": "EPSG:3857"})
    return out


def as_submission(feats: list[dict]) -> list[dict]:
    """Та же последняя цепочка, что пишет файл сдачи. Порог 0 уверенность 1 не режет."""
    from expds.watchdog import finish

    return finish(feats, {"default_threshold": 0.0, "thresholds": {}, "thresholds_by_markup": {}})


def load_metric(path: str):
    spec = importlib.util.spec_from_file_location("oracle_compute_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _score(metric, pred: list[dict], gt: dict) -> dict:
    from expds.writer import to_feature_collection

    return metric.F1Calculator().calculate_f1(to_feature_collection(pred), gt)


def _class_sizes(metric, pred: list[dict], gt: dict) -> dict:
    """Полнота по корзинам площади внутри класса. Точки в корзины не входят."""
    from expds.constants import CLASSES

    out = {}
    for cls in CLASSES:
        sub = {"type": "FeatureCollection",
               "features": [f for f in gt.get("features", [])
                            if (f.get("properties") or {}).get("class_name") == cls]}
        if not sub["features"]:
            continue
        got = [f for f in pred if f["class_name"] == cls]
        sizes = _score(metric, got, sub)["auxiliary"]["size_categories"]
        out[cls] = [{"max_area": row["max_area"], "gt": row["gt"], "tp": row["tp"],
                     "recall": None if row["recall"] is None else round(row["recall"], 4)}
                    for row in sizes if row["gt"]]
    return out


def evaluate(gt: dict, metric, masks_as_circles: bool) -> dict:
    pred = as_submission(emit(gt, masks_as_circles))
    raw = _score(metric, pred, gt)
    per = {}
    for cls, row in raw["per_class"].items():
        if row["num_ground_truth"] == 0:
            continue
        per[cls] = {"recall": round(row["recall"], 4), "f1": round(row["f1"], 4),
                    "tp": row["tp"], "fp": row["fp"], "gt": row["num_ground_truth"],
                    "predictions": row["num_predictions"]}
    sizes = [{"max_area": row["max_area"], "gt": row["gt"], "tp": row["tp"],
              "recall": None if row["recall"] is None else round(row["recall"], 4)}
             for row in raw["auxiliary"]["size_categories"]]
    return {"final_score": round(raw["final_score"], 6), "predictions": len(pred),
            "per_class": per, "size_categories": sizes,
            "size_by_class": _class_sizes(metric, pred, gt)}


def _below(report: dict, bar: float) -> list[str]:
    return [cls for cls, row in report["per_class"].items() if row["recall"] < bar]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", default=str(ROOT / "materials" / "compute_metrics.py"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--bar", type=float, default=0.9, help="полнота ниже этого — дыра геометрии выпуска")
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    metric = load_metric(args.metric)
    report = {"bar": args.bar,
              "shipped": evaluate(gt, metric, False),
              "masks_as_circles": evaluate(gt, metric, True)}
    report["shipped_below_bar"] = _below(report["shipped"], args.bar)
    report["circles_below_bar"] = _below(report["masks_as_circles"], args.bar)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
