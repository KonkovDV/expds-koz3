#!/usr/bin/env python3
"""Окна спутника (SpOr) для детектора YOLO26 / RT-DETR — шаг 1 плана docs/plans/OPTICS_DETECTOR_PLAN_20261105.md.

    python export_spor.py OUT_DIR [--gsd 0.5] [--tile 1024] [--overlap 0.25] [--sites S1,S2] [--empty-keep 0.3]

- Снимки берутся из бакета организаторов (ключи — ~/.secrets/s3_train.env, не печатаются; список ключей — S3_KEYS).
  Каждый GeoTIFF перепроецируется в UTM участка (UTM.json) с шагом --gsd: вниз — average, вверх — bilinear.
- Разметка — gt_train_v2.geojson (EPSG:3857), markup_type sp_or. Рамка — описывающий прямоугольник полигона
  в пикселях окна, не меньше 12 px. Объект берётся в окно, если в окне ≥ 40 % площади его рамки.
- Окна 1024 с перекрытием 25 %. Отбрасываются окна, где вне снимка (alpha = 0 или все каналы 0) > 60 %.
  Пустые окна (без объектов) сохраняются с вероятностью --empty-keep (seed 0).
- Выход (формат ultralytics): images/*.jpg (q95, RGB), labels/*.txt (cls cx cy w h), index.csv, data_f{k}.yaml и
  train_f{k}.txt / val_f{k}.txt по 5 честным фолдам семей (kaggle/2026-10-28/nolidar_verifier/folds_family_k5.json).
- Участок уже выгружен (OUT/done/<участок>) — пропускается. Сырые снимки удаляются сразу после нарезки (диск).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
GT = Path(os.environ.get("KOZ3_GT", "/workspace/koz3/w5data/gt_train_v2.geojson"))
FOLDS = REPO / "kaggle/2026-10-28/nolidar_verifier/folds_family_k5.json"
KEYS = Path(os.environ.get("S3_KEYS", "/tmp/claude-0/-home-user-expds-koz3/da1f6232-979f-5fb0-a9ff-7716188235d3/"
                                      "scratchpad/s3tools/s3_keys.tsv"))


def s3_client():
    import boto3
    from botocore.config import Config
    env = dict(l.strip().split("=", 1) for l in (Path.home() / ".secrets/s3_train.env").read_text().splitlines() if "=" in l)
    c = boto3.client("s3", endpoint_url=env["S3_ENDPOINT"], aws_access_key_id=env["S3_ACCESS_KEY_ID"],
                     aws_secret_access_key=env["S3_SECRET_ACCESS_KEY"], region_name="ru-7",
                     config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8}))
    return c, env["S3_BUCKET"]


def utm_crs(utm_json: dict) -> str:
    import re
    m = re.search(r"EPSG:*\s*(\d+)", str(utm_json.get("crs")), re.I)
    return f"EPSG:{m.group(1)}"


def warp(src_path: Path, crs: str, gsd: float):
    """GeoTIFF → (rgb uint8 [3,H,W], valid bool [H,W], transform) в crs с шагом gsd."""
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    with rasterio.open(src_path) as src:
        tr, W, H = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds, resolution=gsd)
        src_res = abs(src.transform.a) if src.crs and src.crs.is_projected else None
        rs = Resampling.average if (src_res is None or src_res < gsd) else Resampling.bilinear
        nb = min(src.count, 4)
        out = np.zeros((nb, H, W), np.uint8)
        for b in range(nb):
            band = src.read(b + 1)
            if band.dtype != np.uint8:
                hi = np.percentile(band[band > 0], 99.5) if (band > 0).any() else 1.0
                band = np.clip(band.astype(np.float32) / max(hi, 1e-6) * 255, 0, 255).astype(np.uint8)
            reproject(band, out[b], src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=crs,
                      resampling=rs, src_nodata=None, dst_nodata=0)
    rgb = out[:3] if nb >= 3 else np.repeat(out[:1], 3, axis=0)
    valid = (out[3] > 0) if nb >= 4 else (rgb.max(axis=0) > 0)
    return rgb, valid, tr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--sites", default="")
    ap.add_argument("--empty-keep", type=float, default=0.3)
    ap.add_argument("--min-px", type=int, default=12)
    a = ap.parse_args()
    from PIL import Image
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely import transform as _st  # noqa: F401
    from shapely.ops import transform as stf

    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    fold_of = {r: f["fold"] for f in folds["folds"] for r in f["test_regions"]}
    gt = {}
    for f in json.loads(GT.read_text(encoding="utf-8"))["features"]:
        q = f["properties"]
        if q.get("markup_type") == "sp_or" and q.get("class_name") in CLASSES:
            gt.setdefault(q["region_name"], []).append((CLASSES.index(q["class_name"]), shape(f["geometry"])))
    keys = [l.rstrip("\n").split("\t", 2) for l in open(KEYS, encoding="utf-8")]
    tifs: dict = {}
    for size, _, k in keys:
        if "_SpOr_" in k and k.lower().endswith((".tif", ".tiff")) and "_разметка/" not in k:
            tifs.setdefault(k.split("/")[0], []).append(k)
    sites = [s for s in a.sites.split(",") if s] or sorted(tifs)
    for d in ("images", "labels", "done", "raw", "meta"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    s3, bucket = s3_client()
    rng = random.Random(0)
    idx_p = a.out / "index.csv"
    new_idx = not idx_p.exists()
    with open(idx_p, "a", newline="", encoding="utf-8") as fidx:
        w = csv.writer(fidx)
        if new_idx:
            w.writerow(["tile", "site", "fold", "image", "x0", "y0", "n_obj", "gsd"])
        for site in sites:
            if (a.out / "done" / site).exists():
                continue
            if site not in fold_of:
                print(f"{site}: нет в фолдах — пропуск", flush=True)
                continue
            t0 = time.time()
            uj = json.loads(s3.get_object(Bucket=bucket, Key=f"{site}/UTM.json")["Body"].read())
            crs = utm_crs(uj)
            to_utm = Transformer.from_crs("EPSG:3857", crs, always_xy=True).transform
            objs = [(c, stf(to_utm, g)) for c, g in gt.get(site, [])]
            n_tiles = n_obj = 0
            for k in sorted(tifs[site]):
                raw = a.out / "raw" / Path(k).name
                s3.download_file(bucket, k, str(raw))
                try:
                    rgb, valid, tr = warp(raw, crs, a.gsd)
                finally:
                    raw.unlink(missing_ok=True)
                H, W = valid.shape
                inv = ~tr
                # привязка снимка: окно (x0, y0) → мир = tr * (x0 + px, y0 + py), CRS участка (для OOF по метрике)
                (a.out / "meta" / f"{site}__{Path(k).stem}.json").write_text(json.dumps(
                    {"site": site, "image": Path(k).name, "crs": crs, "transform": list(tr)[:6], "H": H, "W": W,
                     "gsd": a.gsd}, ensure_ascii=False), encoding="utf-8")
                boxes = []
                for c, g in objs:
                    x0, y0, x1, y1 = g.bounds
                    (px0, py1), (px1, py0) = inv * (x0, y0), inv * (x1, y1)
                    boxes.append((c, min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1)))
                step = max(1, int(a.tile * (1 - a.overlap)))
                ys = sorted(set(list(range(0, max(1, H - a.tile), step)) + [max(0, H - a.tile)]))
                xs = sorted(set(list(range(0, max(1, W - a.tile), step)) + [max(0, W - a.tile)]))
                for yy in ys:
                    for xx in xs:
                        v = valid[yy:yy + a.tile, xx:xx + a.tile]
                        if v.size == 0 or v.mean() < 0.4:
                            continue
                        lines = []
                        for c, bx0, by0, bx1, by1 in boxes:
                            ix0, iy0 = max(bx0, xx), max(by0, yy)
                            ix1, iy1 = min(bx1, xx + a.tile), min(by1, yy + a.tile)
                            if ix1 <= ix0 or iy1 <= iy0:
                                continue
                            full = max(1e-6, (bx1 - bx0) * (by1 - by0))
                            if (ix1 - ix0) * (iy1 - iy0) / full < 0.4:
                                continue
                            cx, cy = (ix0 + ix1) / 2 - xx, (iy0 + iy1) / 2 - yy
                            bw, bh = max(ix1 - ix0, a.min_px), max(iy1 - iy0, a.min_px)
                            lines.append(f"{c} {cx / a.tile:.6f} {cy / a.tile:.6f} {min(bw, a.tile) / a.tile:.6f} "
                                         f"{min(bh, a.tile) / a.tile:.6f}")
                        if not lines and rng.random() > a.empty_keep:
                            continue
                        tile = np.zeros((a.tile, a.tile, 3), np.uint8)
                        crop = rgb[:, yy:yy + a.tile, xx:xx + a.tile].transpose(1, 2, 0)
                        tile[:crop.shape[0], :crop.shape[1]] = crop
                        name = f"{site}__{Path(k).stem}__{yy}_{xx}"
                        Image.fromarray(tile).save(a.out / "images" / f"{name}.jpg", quality=95)
                        (a.out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                    encoding="utf-8")
                        w.writerow([name, site, fold_of[site], Path(k).name, xx, yy, len(lines), a.gsd])
                        n_tiles += 1
                        n_obj += len(lines)
                del rgb, valid
            (a.out / "done" / site).write_text(json.dumps({"tiles": n_tiles, "objects": n_obj}), encoding="utf-8")
            fidx.flush()
            print(f"{site} f{fold_of[site]}: окон {n_tiles}, объектов {n_obj}, {time.time() - t0:.0f} с", flush=True)
    rows = list(csv.DictReader(open(idx_p, encoding="utf-8")))
    for k in range(5):
        tr_ = [f"./images/{r['tile']}.jpg" for r in rows if int(r["fold"]) != k]
        va_ = [f"./images/{r['tile']}.jpg" for r in rows if int(r["fold"]) == k]
        (a.out / f"train_f{k}.txt").write_text("\n".join(tr_) + "\n", encoding="utf-8")
        (a.out / f"val_f{k}.txt").write_text("\n".join(va_) + "\n", encoding="utf-8")
        (a.out / f"data_f{k}.yaml").write_text(
            f"path: .\ntrain: train_f{k}.txt\nval: val_f{k}.txt\nnames:\n" +
            "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    (a.out / "train_all.txt").write_text("\n".join(f"./images/{r['tile']}.jpg" for r in rows) + "\n", encoding="utf-8")
    print(json.dumps({"tiles": len(rows), "objects": sum(int(r["n_obj"]) for r in rows),
                      "sites": len({r["site"] for r in rows})}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
