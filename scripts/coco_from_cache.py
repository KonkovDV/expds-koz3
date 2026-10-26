"""Кэш тайлов → COCO. Детектор не учит и в сдачу не входит.

Кэш собран вокруг объектов. Инференс U-Net идёт сплошной сеткой. Рамки с этого
кэша годятся, чтобы учить детектор и проверить перевод рамки в круг, и не годятся
как единственное доказательство, что детектор лучше карты центров. Для оценки
нужен кэш сплошной сеткой той же семьи (`python -m expds.tiles ... --grid-overlap 0.25`)
и обратный перевод предсказаний в стек (`scripts/coco_preds_to_stack.py`).

    # только разметка, как до 27.10
    python scripts/coco_from_cache.py --cache C --out O --val-families Альфа Бета
    # папки RF-DETR: train/valid/test с _annotations.coco.json и PNG, фолд 0 из 5 по семьям
    python scripts/coco_from_cache.py --cache C --out O --layout rfdetr --folds 5 --fold 0
    # проверочный кэш сплошной сеткой целиком в valid/test
    python scripts/coco_from_cache.py --cache C_grid --out O_grid --layout rfdetr --all-valid

В каждой записи `images` — центр окна, шаг, сторона и CRS источника (`crs_by_source`
из meta.json кэша): по ним рамка возвращается на землю. Номера классов 1…K в порядке
`meta["classes"]`. PNG — три канала uint8 в порядке `--channels`: детектор с весами
COCO ждёт RGB, лишние каналы кэша отбрасываются, недостающие повторяют последний.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds.det_boxes import boxes_from_mask
from expds.tiles import IGNORE_BIT


def image_name(row: dict) -> str:
    return f"{int(row['shard']):04d}_{int(row['idx']):06d}.png"


def coco_from_rows(cache: Path, rows: list[dict], classes: list[str], ignore_bit: int,
                   meta: dict | None = None) -> dict:
    meta = meta or {}
    crs_map = meta.get("crs_by_source") or {}
    images, annotations = [], []
    ann_id = 1
    for image_id, row in enumerate(rows, start=1):
        masks = np.load(cache / f"masks_{int(row['shard']):04d}.npy", mmap_mode="r")
        boxes = boxes_from_mask(masks[int(row["idx"])], classes, ignore_bit)
        images.append({
            "id": image_id,
            "file_name": image_name(row),
            "family": row["family"],
            "region": row["region"],
            "source": row["file"],
            "cx": float(row["cx"]),
            "cy": float(row["cy"]),
            "res": meta.get("res"),
            "tile": meta.get("tile"),
            "markup": row.get("markup") or meta.get("markup"),
            "crs": crs_map.get(f"{row['region']}|{row['file']}"),
            "width": meta.get("tile"),
            "height": meta.get("tile"),
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
        "categories": [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(classes)],
    }


def to_rgb(img: np.ndarray, channels: list[int]) -> np.ndarray:
    """C×H×W uint8 → H×W×3 в порядке channels; номер вне кэша повторяет последний канал."""
    c = img.shape[0]
    picked = [img[min(k, c - 1)] for k in channels[:3]]
    while len(picked) < 3:
        picked.append(picked[-1])
    return np.ascontiguousarray(np.stack(picked, axis=-1))


def write_images(cache: Path, rows: list[dict], folder: Path, channels: list[int]) -> int:
    import cv2

    folder.mkdir(parents=True, exist_ok=True)
    shards: dict[int, np.ndarray] = {}
    n = 0
    for row in rows:
        s = int(row["shard"])
        if s not in shards:
            shards[s] = np.load(cache / f"images_{s:04d}.npy", mmap_mode="r")
        rgb = to_rgb(np.asarray(shards[s][int(row["idx"])]), channels)
        ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])
        if not ok:
            raise OSError(f"PNG не записан: {folder / image_name(row)}")
        (folder / image_name(row)).write_bytes(buf.tobytes())
        n += 1
    return n


def _link_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if target.exists():
            target.unlink()
        try:
            os.link(f, target)
        except OSError:
            shutil.copy2(f, target)


def split_rows(rows: list[dict], val_families: list[str] | None, folds: int | None, fold: int | None,
               all_valid: bool = False) -> dict[str, list[dict]]:
    if all_valid:
        return {"train": [], "valid": list(rows)}
    if val_families:
        val = set(val_families)
    elif folds:
        from expds.metric import family_folds
        from expds.naming import site_family

        regions = sorted({r["region"] for r in rows})
        parts = family_folds(regions, site_family, folds)
        if fold is None or not 0 <= fold < len(parts):
            raise SystemExit(f"--fold от 0 до {len(parts) - 1}")
        held = parts[fold]
        return {"train": [r for r in rows if r["region"] not in held],
                "valid": [r for r in rows if r["region"] in held]}
    else:
        return {"all": list(rows)}
    return {"train": [r for r in rows if r["family"] not in val],
            "valid": [r for r in rows if r["family"] in val]}


def export(cache: Path, out: Path, layout: str = "json", images: bool = False,
           val_families: list[str] | None = None, folds: int | None = None, fold: int | None = None,
           channels: list[int] | None = None, all_valid: bool = False) -> dict:
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    classes = list(meta["classes"])
    ignore_bit = int(meta.get("ignore_bit", IGNORE_BIT))
    with (cache / "index.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    parts = split_rows(rows, val_families, folds, fold, all_valid)
    channels = channels or [0, 1, 2]
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"classes": classes, "parts": {}}
    for name, part in parts.items():
        coco = coco_from_rows(cache, part, classes, ignore_bit, meta)
        if layout == "rfdetr":
            folder = out / name
            if images:
                write_images(cache, part, folder, channels)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "_annotations.coco.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        else:
            fname = {"all": "all.json", "train": "train.json", "valid": "val.json"}[name]
            (out / fname).write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
            if images:
                write_images(cache, part, out / "images", channels)
        summary["parts"][name] = {"images": len(coco["images"]), "boxes": len(coco["annotations"])}
    if layout == "rfdetr" and "valid" in parts:
        # RF-DETR ждёт и test; это та же проверка, не третья выборка.
        _link_dir(out / "valid", out / "test")
    if not meta.get("crs_by_source"):
        summary["warning"] = "в meta.json нет crs_by_source: кэш собран до 27.10, рамку на землю не вернуть"
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-families", nargs="*", default=None,
                    help="семьи проверки; без них и без --folds пишется один файл на все тайлы")
    ap.add_argument("--folds", type=int, default=None, help="фолды по семьям (expds.metric.family_folds)")
    ap.add_argument("--fold", type=int, default=None, help="номер фолда проверки при --folds")
    ap.add_argument("--all-valid", action="store_true", help="все тайлы в valid/test (кэш сплошной сеткой)")
    ap.add_argument("--layout", choices=["json", "rfdetr"], default="json",
                    help="json — файлы разметки как до 27.10; rfdetr — папки train/valid/test")
    ap.add_argument("--images", action="store_true", help="записать PNG (для rfdetr обязательно)")
    ap.add_argument("--channels", default="0,1,2", help="каналы кэша в RGB, через запятую")
    args = ap.parse_args()
    images = args.images or args.layout == "rfdetr"
    summary = export(Path(args.cache), Path(args.out), args.layout, images, args.val_families, args.folds,
                     args.fold, [int(c) for c in args.channels.split(",")], args.all_valid)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
