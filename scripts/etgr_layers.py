#!/usr/bin/env python3
"""Попадания object_poly электроразведки отдельно на картах срезов и на профилях.

Имя файла решает вид: «карта» — срез, «профил» — профиль. Остальное в сводку не входит.
Имена участков не печатаются.

    python scripts/etgr_layers.py --pred F:/expds3/work/geo_cur.geojson \
        --gt F:/expds3/work/gt_train_v2.geojson \
        --metric C:/Sloi/files/materials/compute_metrics.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def layer_kind(name: str) -> str | None:
    """«map», «profile» или None, если по имени вид слоя не виден."""
    low = (name or "").lower().replace("ё", "е")
    if "профил" in low or "profile" in low:
        return "profile"
    if "карта" in low or "карт_" in low:
        return "map"
    return None


def _source(feat: dict) -> str:
    props = feat.get("properties") or {}
    return str(props.get("source") or props.get("file") or feat.get("source") or "")


def _etgr(fc: dict) -> list[dict]:
    out = []
    for feat in fc.get("features", []):
        props = feat.get("properties") or {}
        if props.get("markup_type") == "etgr" and props.get("class_name") == "object_poly":
            out.append(feat)
    return out


def subset(fc: dict, kind: str) -> dict:
    feats = [f for f in _etgr(fc) if layer_kind(_source(f)) == kind]
    return {"type": "FeatureCollection", "features": feats}


def predictions_for(pred: dict, kind: str) -> dict:
    """Слой предсказания, если имя файла сохранилось. Иначе все объекты EtGr.

    Записанный geojson сдачи имя файла не хранит, поэтому один и тот же набор
    кругов сравнивается с эталоном карт и отдельно с эталоном профилей.
    """
    tagged = subset(pred, kind)
    if tagged["features"] or any(layer_kind(_source(f)) for f in _etgr(pred)):
        return tagged
    return {"type": "FeatureCollection", "features": _etgr(pred)}


def score_kind(metric, pred: dict, gt: dict, kind: str) -> dict:
    got = predictions_for(pred, kind)
    truth = subset(gt, kind)
    raw = metric.F1Calculator().calculate_f1(got, truth)
    row = raw["per_class"].get("object_poly") or {}
    return {"kind": kind, "pred": len(got["features"]), "gt": len(truth["features"]),
            "tp": row.get("tp", 0), "fp": row.get("fp", 0),
            "recall": None if not row else round(row.get("recall", 0.0), 4),
            "f1": None if not row else round(row.get("f1", 0.0), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("etgr_metric", args.metric)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    report = {"layers": [score_kind(mod, pred, gt, kind) for kind in ("map", "profile")]}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
