#!/usr/bin/env python3
"""Близость участков выборки к участкам трейна: те же памятники или новые места.

Если участки публичной (и, вероятно, закрытой) выборки лежат в нескольких сотнях метров
от участков трейна, выборка — не «новые семьи», а соседние планшеты тех же памятников.
Тогда модели на всех данных и пороги по всем участкам переносятся лучше, чем обещает
проверка по семьям, а классика геофизики с параметрами «на новые места» отдаёт балл
обученной сети.

    python scripts/site_overlap.py --train F:/expds3/train-expds-3 --val <папка публичной выборки> \
        --out notes/site_overlap_public.json

Имена участков выборки в вывод не попадают (номера по порядку). Имена участков трейна —
те же, что уже в notes/. Расстояние — между рамками в градусах, пересчитанное в метры
(для рамок до десятков километров этого хватает).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds.catalog import read_utm_json, scan_root
from expds.crs import las_box, las_crs, valid_crs
from expds.naming import site_family


def raster_lonlat_box(path: Path) -> tuple | None:
    """Рамка растра в EPSG:4326 или None (нет CRS)."""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as ds:
        if ds.crs is None:
            return None
        return tuple(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=5))


def las_lonlat_box(path: Path, region_crs: str | None) -> tuple | None:
    """Рамка облака в EPSG:4326 по заголовку LAS и CRS участка."""
    from pyproj import Transformer

    crs = valid_crs(region_crs) or valid_crs(las_crs(path))
    box = las_box(path)
    if not crs or box is None:
        return None
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs, ys = tr.transform([box[0], box[2], box[0], box[2]], [box[1], box[1], box[3], box[3]])
    return min(xs), min(ys), max(xs), max(ys)


def region_box(reg) -> tuple[tuple | None, list[str]]:
    """Общая рамка всех привязанных файлов участка и список его типов данных."""
    utm = read_utm_json(reg.utm_json)
    boxes = []
    for files in reg.rasters.values():
        for p in files:
            try:
                b = raster_lonlat_box(p)
            except Exception:
                b = None
            if b and all(map(math.isfinite, b)):
                boxes.append(b)
    for p in reg.las:
        try:
            b = las_lonlat_box(p, utm)
        except Exception:
            b = None
        if b and all(map(math.isfinite, b)):
            boxes.append(b)
    if not boxes:
        return None, reg.markup_types
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)), reg.markup_types


def gap_m(a: tuple, b: tuple) -> float:
    """Расстояние между рамками (0 — пересекаются), метры."""
    lat = math.radians((a[1] + a[3] + b[1] + b[3]) / 4)
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2])) * 111_320 * math.cos(lat)
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3])) * 110_574
    return math.hypot(dx, dy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = []
    for reg in scan_root(args.train):
        box, types = region_box(reg)
        if box:
            train.append((reg.name, site_family(reg.name), box, types))
    rows = []
    for i, reg in enumerate(scan_root(args.val)):
        box, types = region_box(reg)
        rec = {"region": f"val_{i:03d}", "types": types}
        if box is None or not train:
            rec["note"] = "нет привязанных файлов"
            rows.append(rec)
            continue
        dist = sorted((gap_m(box, t[2]), t) for t in train)
        d, near = dist[0]
        rec.update({"nearest_train": near[0], "family": near[1], "gap_m": round(d, 1),
                    "overlaps": d == 0.0, "train_types": near[3],
                    "within_5km": sorted({t[1] for g, t in dist if g <= 5000})})
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    placed = [r for r in rows if "gap_m" in r]
    if placed:
        summary = {f"<= {lim} m": sum(r["gap_m"] <= lim for r in placed) for lim in (0, 500, 2000, 10000, 50000)}
        by_type = {}
        for r in placed:
            for t in r["types"]:
                by_type.setdefault(t, []).append(r["gap_m"])
        summary["median_gap_m_by_type"] = {t: sorted(v)[len(v) // 2] for t, v in sorted(by_type.items())}
        print(json.dumps({"regions": len(rows), "placed": len(placed), **summary}, ensure_ascii=False))
        rows.append({"summary": summary})
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
