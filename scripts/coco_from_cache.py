"""Кэш тайлов → COCO. Детектор не учит и в сдачу не входит.

Кэш собран вокруг объектов. Инференс U-Net идёт сплошной сеткой. Рамки с этого
кэша годятся, чтобы проверить перевод рамки в круг, и не годятся как единственное
доказательство, что детектор лучше карты центров.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from expds.det_boxes import boxes_from_mask
from expds.tiles import IGNORE_BIT


def coco_from_rows(cache: Path, rows: list[dict], classes: list[str], ignore_bit: int) -> dict:
    images, annotations = [], []
    ann_id = 1
    for image_id, row in enumerate(rows, start=1):
        masks = np.load(cache / f"masks_{int(row['shard']):04d}.npy", mmap_mode="r")
        boxes = boxes_from_mask(masks[int(row["idx"])], classes, ignore_bit)
        images.append({
            "id": image_id,
            "file_name": f"{int(row['shard']):04d}_{int(row['idx']):06d}.png",
            "family": row["family"],
            "region": row["region"],
            "source": row["file"],
            "cx": float(row["cx"]),
            "cy": float(row["cy"]),
        })
        for box in boxes:
            x, y, w, h = box["bbox"]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": classes.index(box["category"]) + 1,
                "bbox": box["bbox"],
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(classes)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Экспорт кэша тайлов в COCO без обучения детектора")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-families", nargs="*", default=None,
                    help="семьи проверки; без аргумента пишется один файл на все тайлы")
    args = ap.parse_args()
    cache = Path(args.cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    classes = list(meta["classes"])
    ignore_bit = int(meta.get("ignore_bit", IGNORE_BIT))
    with (cache / "index.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.val_families:
        (out / "all.json").write_text(
            json.dumps(coco_from_rows(cache, rows, classes, ignore_bit), ensure_ascii=False),
            encoding="utf-8")
        return
    val = set(args.val_families)
    parts = {"train": [r for r in rows if r["family"] not in val],
             "val": [r for r in rows if r["family"] in val]}
    for name, part in parts.items():
        (out / f"{name}.json").write_text(
            json.dumps(coco_from_rows(cache, part, classes, ignore_bit), ensure_ascii=False),
            encoding="utf-8")


if __name__ == "__main__":
    main()
