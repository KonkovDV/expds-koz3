#!/usr/bin/env python3
"""Окна немецкой аэрофотосъёмки для детектора. Снимки уже на диске, бакет не трогаем.

    python export_ae.py OUT --root /workspace/data/train-expds-3

Шаг 0,5 м, окно 1024, перекрытие 25 %. Класс берётся из имени файла разметки.
Пустые окна сохраняются с вероятностью --empty-keep. Тимерево — 498 меток на одном
листе: потолок детекций на инференсе должен быть выше 300.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from expds.naming import class_of_annotation  # noqa: E402

CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
FOLDS = REPO / "kaggle/2026-10-28/nolidar_verifier/folds_family_k5.json"


def utm_crs(utm_json: dict) -> str:
    import re
    m = re.search(r"EPSG:*\s*(\d+)", str(utm_json.get("crs")), re.I)
    if not m:
        raise ValueError("в UTM.json нет EPSG")
    return f"EPSG:{m.group(1)}"


def is_ae_name(name: str) -> bool:
    low = unicodedata.normalize("NFC", name).lower()
    return "_ae_" in low or "_ае_" in low or "аэро" in low


def warp(src_path: Path, crs: str, gsd: float):
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    with rasterio.open(src_path) as src:
        tr, W, H = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds, resolution=gsd)
        src_res = abs(src.transform.a)
        rs = Resampling.average if src_res < gsd else Resampling.bilinear
        band = src.read(1)
        if band.dtype != np.uint8:
            hi = np.percentile(band[np.isfinite(band) & (band > 0)], 99.5) if (band > 0).any() else 1.0
            band = np.clip(band.astype(np.float32) / max(float(hi), 1e-6) * 255, 0, 255).astype(np.uint8)
        out = np.zeros((H, W), np.uint8)
        reproject(band, out, src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=crs,
                  resampling=rs, dst_nodata=0)
    valid = out > 0
    rgb = np.repeat(out[None], 3, axis=0)
    return rgb, valid, tr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--root", type=Path, default=Path("/workspace/data/train-expds-3"))
    ap.add_argument("--gsd", type=float, default=0.5)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--empty-keep", type=float, default=0.2)
    ap.add_argument("--min-px", type=int, default=8)
    a = ap.parse_args()
    from PIL import Image
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as stf

    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    fold_of = {r: f["fold"] for f in folds["folds"] for r in f["test_regions"]}
    for d in ("images", "labels", "done", "meta"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    idx_p = a.out / "index.csv"
    sites = sorted(p for p in a.root.iterdir() if p.is_dir() and any(is_ae_name(t.name) for t in p.rglob("*.tif")))
    with open(idx_p, "w", newline="", encoding="utf-8") as fidx:
        w = csv.writer(fidx)
        w.writerow(["tile", "site", "fold", "image", "x0", "y0", "n_obj", "gsd"])
        for site_dir in sites:
            site = site_dir.name
            if site not in fold_of:
                print(f"{site}: нет в фолдах — пропуск", flush=True)
                continue
            t0 = time.time()
            uj = json.loads((site_dir / "UTM.json").read_text(encoding="utf-8"))
            crs = utm_crs(uj)
            to_utm = Transformer.from_crs("EPSG:3857", crs, always_xy=True).transform
            objs = []
            for gj in site_dir.rglob("*.geojson"):
                if not is_ae_name(gj.name):
                    continue
                cls = class_of_annotation(gj)
                if cls not in CLASSES:
                    continue
                c = CLASSES.index(cls)
                data = json.loads(gj.read_text(encoding="utf-8"))
                for feat in data.get("features") or []:
                    geom = feat.get("geometry")
                    if not geom:
                        continue
                    objs.append((c, stf(to_utm, shape(geom))))
            n_tiles = n_obj = 0
            tifs = sorted(t for t in site_dir.rglob("*.tif") if is_ae_name(t.name))
            for src in tifs:
                rgb, valid, tr = warp(src, crs, a.gsd)
                H, W = valid.shape
                inv = ~tr
                (a.out / "meta" / f"{site}__{src.stem}.json").write_text(json.dumps(
                    {"site": site, "image": src.name, "crs": crs, "transform": list(tr)[:6],
                     "H": H, "W": W, "gsd": a.gsd}, ensure_ascii=False), encoding="utf-8")
                boxes = []
                for c, g in objs:
                    if g.is_empty:
                        continue
                    x0, y0, x1, y1 = g.bounds
                    (px0, py1), (px1, py0) = inv * (x0, y0), inv * (x1, y1)
                    boxes.append((c, min(px0, px1), min(py0, py1), max(px0, px1), max(py0, py1)))
                step = max(1, int(a.tile * (1 - a.overlap)))
                ys = sorted(set(list(range(0, max(1, H - a.tile), step)) + [max(0, H - a.tile)]))
                xs = sorted(set(list(range(0, max(1, W - a.tile), step)) + [max(0, W - a.tile)]))
                for yy in ys:
                    for xx in xs:
                        v = valid[yy:yy + a.tile, xx:xx + a.tile]
                        if v.size == 0 or float(v.mean()) < 0.4:
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
                            lines.append(f"{c} {cx / a.tile:.6f} {cy / a.tile:.6f} "
                                         f"{min(bw, a.tile) / a.tile:.6f} {min(bh, a.tile) / a.tile:.6f}")
                        if not lines and rng.random() > a.empty_keep:
                            continue
                        tile = np.zeros((a.tile, a.tile, 3), np.uint8)
                        crop = rgb[:, yy:yy + a.tile, xx:xx + a.tile].transpose(1, 2, 0)
                        tile[:crop.shape[0], :crop.shape[1]] = crop
                        name = f"{site}__{src.stem}__{yy}_{xx}"
                        Image.fromarray(tile).save(a.out / "images" / f"{name}.jpg", quality=90)
                        (a.out / "labels" / f"{name}.txt").write_text(
                            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                        w.writerow([name, site, fold_of[site], src.name, xx, yy, len(lines), a.gsd])
                        n_tiles += 1
                        n_obj += len(lines)
                del rgb, valid
            (a.out / "done" / site).write_text(json.dumps({"tiles": n_tiles, "objects": n_obj}), encoding="utf-8")
            fidx.flush()
            print(f"{site} f{fold_of[site]}: окон {n_tiles}, рамок {n_obj}, меток {len(objs)}, {time.time()-t0:.0f} с",
                  flush=True)
    rows = list(csv.DictReader(open(idx_p, encoding="utf-8")))
    for k in range(5):
        tr_ = [f"./images/{r['tile']}.jpg" for r in rows if int(r["fold"]) != k]
        va_ = [f"./images/{r['tile']}.jpg" for r in rows if int(r["fold"]) == k]
        (a.out / f"train_f{k}.txt").write_text("\n".join(tr_) + "\n", encoding="utf-8")
        (a.out / f"val_f{k}.txt").write_text("\n".join(va_) + "\n", encoding="utf-8")
        (a.out / f"data_f{k}.yaml").write_text(
            "path: .\ntrain: train_f{k}.txt\nval: val_f{k}.txt\nnames:\n".format(k=k) +
            "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
    print(json.dumps({"tiles": len(rows), "boxes": sum(int(r["n_obj"]) for r in rows),
                      "sites": len({r["site"] for r in rows})}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
