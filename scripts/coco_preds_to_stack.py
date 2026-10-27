#!/usr/bin/env python3
"""Предсказания детектора рамок (формат результатов COCO) → круги стека для табло.

Детектор (например, RF-DETR) предсказывает рамки на PNG из `scripts/coco_from_cache.py`.
Для честного сравнения с U-Net это должен быть кэш сплошной сеткой (`--grid-overlap`) по
участкам проверочного фолда: тогда круги покрывают участок так же, как на инференсе.
Каждая рамка становится кругом в своём центре (`expds.det_boxes.circle_from_box`), как
пик карты центров: метрике нужна точка внутри эталона, контур на балл не влияет.

Окна сетки перекрываются, и один объект виден в двух-четырёх окнах. Дубли снимаются по
участку и классу: кандидаты идут по убыванию уверенности; кандидат, чей центр ближе `--merge-m`
к рамке кучки или чья рамка на земле хотя бы наполовину (от меньшей площади, как в метрике)
лежит в рамке кучки, присоединяется к ней. Обрезанная краем окна рамка — часть целой, поэтому
так и склеивается, хотя её центр сдвинут. Кучка выдаёт один круг с наибольшей уверенностью;
круг ставится по целой рамке, если она есть.

    python scripts/coco_preds_to_stack.py --coco O_grid/valid/_annotations.coco.json \
        --preds rfdetr_valid.json --group sp_rfdetr --radius 4 --out F:/expds3/work/sp_rfdetr_f0.geojson
    python scripts/scoreboard.py <флаги раздела 0> --replace sp_focal=F:/expds3/work/sp_rfdetr.geojson \
        --honest 10

Формат --preds — список {"image_id", "category_id", "bbox": [x, y, w, h], "score"} в пикселях PNG,
как пишет `pycocotools` и большинство детекторов. Номера классов — как в `categories` файла --coco.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def touches_edge(bbox: list[float], tile: int, margin_px: float = 1.0) -> bool:
    x, y, w, h = (float(v) for v in bbox)
    return x <= margin_px or y <= margin_px or x + w >= tile - margin_px or y + h >= tile - margin_px


def _cover(a: tuple, b: tuple) -> float:
    """Площадь пересечения рамок от меньшей из двух (как покрытие в метрике)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / small if small > 0 else 0.0


def ground_per_unit(crs: str | None, y: float) -> float:
    """Метров на земле в одной единице CRS. У UTM это 1, у EPSG:3857 — cos(широты)."""
    if crs and str(crs).upper().endswith(":3857"):
        from expds.open_ideas import mercator_scale

        return 1.0 / mercator_scale(y)
    return 1.0


def ground_gap(a: dict, b: dict) -> float:
    """Расстояние центров рамок в метрах на земле, не в единицах карты."""
    unit = ground_per_unit(a["im"].get("crs"), (a["y"] + b["y"]) / 2.0)
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) * unit


def same_object(a: dict, b: dict, merge_m: float) -> bool:
    return ground_gap(a, b) <= merge_m or _cover(a["box"], b["box"]) >= 0.5


def convert(coco: dict, preds: list[dict], group: str, radius: float, merge_m: float,
            min_score: float = 0.0, radius_by_class: dict[str, float] | None = None) -> dict:
    """Рамки → круги EPSG:3857 с полями стека; дубли окон склеены по участку и классу."""
    from expds.det_boxes import box_center_world, circle_from_box, window_origin

    images = {int(im["id"]): im for im in coco["images"]}
    names = {int(c["id"]): c["name"] for c in coco["categories"]}
    radius_by_class = radius_by_class or {}
    cand: dict[tuple, list[dict]] = defaultdict(list)
    missing_crs = 0
    for p in preds:
        score = float(p.get("score", 0.0))
        if score < min_score:
            continue
        im = images.get(int(p["image_id"]))
        cls = names.get(int(p["category_id"]))
        if im is None or cls is None:
            continue
        if not im.get("crs"):
            missing_crs += 1
            continue
        res, tile = float(im["res"]), int(im["tile"])
        x, y = box_center_world(p["bbox"], im["cx"], im["cy"], res, tile)
        west, north = window_origin(im["cx"], im["cy"], res, tile)
        bx, by, bw, bh = (float(v) for v in p["bbox"])
        world = (west + bx * res, north - (by + bh) * res, west + (bx + bw) * res, north - by * res)
        cand[(im["region"], cls)].append({"x": x, "y": y, "score": score, "bbox": p["bbox"], "im": im,
                                          "edge": touches_edge(p["bbox"], tile), "box": world})
    if missing_crs:
        raise SystemExit(f"{missing_crs} рамок без CRS окна: кэш собран до 27.10, пересоберите его")
    feats = []
    for (region, cls), items in sorted(cand.items()):
        items.sort(key=lambda c: -c["score"])
        clusters: list[list[dict]] = []
        for c in items:
            for cl in clusters:
                if any(same_object(c, m, merge_m) for m in cl):
                    cl.append(c)
                    break
            else:
                clusters.append([c])
        for cl in clusters:
            whole = [c for c in cl if not c["edge"]]
            best = whole[0] if whole else cl[0]
            im = best["im"]
            res, tile = float(im["res"]), int(im["tile"])
            f = circle_from_box(best["bbox"], im["cx"], im["cy"], res, tile,
                                radius_by_class.get(cls, radius), im["crs"], cls, max(c["score"] for c in cl))
            unit = ground_per_unit(im.get("crs"), best["y"])
            w_m, h_m = float(best["bbox"][2]) * res * unit, float(best["bbox"][3]) * res * unit
            feats.append({"type": "Feature", "geometry": f["geometry"].__geo_interface__,
                          "properties": {"class_name": cls, "confidence": round(f["confidence"], 6),
                                         "region_name": region, "markup_type": im.get("markup") or "",
                                         "threshold_group": group, "box_w_m": round(w_m, 2),
                                         "box_h_m": round(h_m, 2), "views": len(cl)}})
    return {"type": "FeatureCollection", "features": feats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", required=True, help="_annotations.coco.json проверочных окон (с cx, cy, res, tile, crs)")
    ap.add_argument("--preds", required=True, help="результаты детектора в формате COCO")
    ap.add_argument("--group", required=True, help="группа порога табло, например sp_rfdetr")
    ap.add_argument("--radius", type=float, default=4.0, help="радиус круга, м на земле (спутник 4, лидар 1,5)")
    ap.add_argument("--radius-class", action="append", default=[], help="класс=радиус, м")
    ap.add_argument("--merge-m", type=float, default=4.0, help="дубли окон ближе этого, м — один объект")
    ap.add_argument("--min-score", type=float, default=0.0, help="рамки слабее не выдаются (порог подбирает табло)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    preds = json.loads(Path(args.preds).read_text(encoding="utf-8"))
    by_class = {k: float(v) for k, v in (s.split("=", 1) for s in args.radius_class)}
    out = convert(coco, preds, args.group, args.radius, args.merge_m, args.min_score, by_class)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"features": len(out["features"]), "group": args.group}, ensure_ascii=False))


if __name__ == "__main__":
    main()
