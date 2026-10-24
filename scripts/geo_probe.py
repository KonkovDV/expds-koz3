#!/usr/bin/env python3
"""Паспорт растров геофизики: что видит детектор на трейне и на публичной выборке.

Эталон не нужен для публичной выборки: скрипт печатает каналы, тип, CRS,
долю пустого фона и сколько пиков находит детектор при разных нижних
отсечках (в MAD). Если дать эталон трейна, для каждой карты MagGr добавляется
потолок полноты находок: доля эталонных точек, у которых ближе 2 м есть пик
выше отсечки, и тот же потолок при радиусе подавления пиков 2 м, как в сдаче.

    python scripts/geo_probe.py --root F:/expds3/train-expds-3 --gt F:/expds3/work/gt_train_v2.geojson --out notes/geo_probe_train.json
    python scripts/geo_probe.py --root F:/expds3/public_val --out notes/geo_probe_public.json

Имена участков публичной выборки в notes/ не переносятся: --anonymize заменяет
их номерами (по умолчанию включено для корня без разметки).

На публичной выборке 24.10 геофизика выдала 570 объектов на 1441 эталон
(65 % попаданий, recall 0,26). Здесь видно, держит ли полноту отсечка 1 MAD,
цветная карта, прочитанная первым каналом, или растр без привязки.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import geophys
from expds.catalog import read_utm_json, scan_root
from expds.raster import pixel_to_xy

FLOORS = (0.5, 0.75, 1.0, 1.5, 2.0)


def raster_passport(path: Path) -> dict:
    """Каналы, тип, CRS, привязка и доля фона без чтения всего растра в память."""
    import rasterio

    with rasterio.open(path) as ds:
        f = max(1, int(np.ceil(max(ds.width, ds.height) / 2048)))
        n = min(ds.count, 4)
        a = ds.read(list(range(1, n + 1)), out_shape=(n, max(1, ds.height // f), max(1, ds.width // f)))
        rgb = a[: min(3, n)].astype(np.float32)
        differ = float(np.mean(np.any(rgb != rgb[:1], axis=0))) if rgb.shape[0] > 1 else 0.0
        vals, cnt = np.unique(a[0], return_counts=True)
        return {
            "bands": ds.count, "dtype": ds.dtypes[0], "nodata": ds.nodata,
            "colorinterp": [getattr(c, "name", str(c)) for c in ds.colorinterp],
            "size": [ds.width, ds.height], "res": round(float(abs(ds.res[0])), 4),
            "crs": ds.crs.to_string() if ds.crs else None,
            "identity_transform": ds.transform.is_identity,
            "channels_differ": round(differ, 4),
            "top_value_share": round(float(cnt.max() / cnt.sum()), 4),
            "top_value": float(vals[int(np.argmax(cnt))]),
        }


def peaks_at_floors(path: Path, mode: str, background_m: float, radius_m: float) -> tuple[dict, tuple]:
    """Число пиков |высоких частот| выше каждой отсечки и сами пики самой низкой."""
    a, tr, crs, res = geophys.read_band(path, mode)
    valid = np.isfinite(a)
    area_m2 = float(valid.sum()) * res * res
    z = geophys.robust_z(a)
    hp = geophys.highpass(z, res, background_m)
    mag = np.where(np.isfinite(hp), np.abs(hp), np.nan).astype(np.float32)
    rows, cols, vals = geophys.accel.find_peaks(mag, max(1, round(radius_m / res)), min(FLOORS))
    xs, ys = pixel_to_xy(tr, rows, cols)
    counts = {str(k): int((vals >= k).sum()) for k in FLOORS}
    dens = {str(k): round(counts[str(k)] / max(area_m2, 1e-9) * 1000.0, 3) for k in FLOORS}
    return {"area_m2": round(area_m2, 1), "peaks": counts, "peaks_per_1000m2": dens}, (xs, ys, vals, crs, hp, tr, res)


def gt_ceiling(gt_xy: np.ndarray, peaks: tuple, tol_m: float = 2.0) -> dict:
    """Доля эталонных точек, у которых ближе tol_m есть пик выше отсечки (после подавления)."""
    from scipy.spatial import cKDTree

    xs, ys, vals, *_ = peaks
    out = {}
    if gt_xy.size == 0:
        return out
    for k in FLOORS:
        keep = vals >= k
        if not keep.any():
            out[str(k)] = 0.0
            continue
        tree = cKDTree(np.c_[xs[keep], ys[keep]])
        d, _ = tree.query(gt_xy, k=1)
        out[str(k)] = round(float(np.mean(d <= tol_m)), 4)
    return out


def gt_strength(gt_xy: np.ndarray, peaks: tuple, tol_m: float = 2.0) -> dict:
    """Сила |высоких частот| у эталонных точек (максимум в круге tol_m): квантили в MAD."""
    *_, _crs, hp, tr, res = peaks
    if gt_xy.size == 0:
        return {}
    inv = ~tr
    r = max(1, round(tol_m / res))
    best = []
    for x, y in gt_xy:
        c, rr = inv * (float(x), float(y))
        c, rr = int(np.floor(c)), int(np.floor(rr))
        win = hp[max(0, rr - r): rr + r + 1, max(0, c - r): c + r + 1]
        win = np.abs(win[np.isfinite(win)]) if win.size else np.array([])
        best.append(float(win.max()) if win.size else np.nan)
    v = np.asarray(best)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {"n": int(v.size), **{f"p{q}": round(float(np.percentile(v, q)), 3) for q in (10, 25, 50, 75, 90)},
            **{f"share_ge_{k}": round(float(np.mean(v >= k)), 4) for k in FLOORS}}


def gt_points(gt: dict, region: str, markup: str, crs: str | None) -> np.ndarray:
    """Эталонные finds_points участка в CRS растра (из EPSG:3857 эталона)."""
    from shapely.geometry import shape

    from expds.crs import WEB_MERCATOR, to_crs

    pts = []
    for f in gt.get("features", []):
        p = f.get("properties") or {}
        if p.get("region_name") != region or p.get("markup_type") != markup or p.get("class_name") != "finds_points":
            continue
        g = shape(f["geometry"])
        if crs:
            g = to_crs(g, WEB_MERCATOR, crs)
        c = g.centroid
        pts.append((c.x, c.y))
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt", default=None, help="эталон трейна в EPSG:3857 (gt_train_v2.geojson)")
    ap.add_argument("--band-mode", default="first", choices=geophys.BAND_MODES)
    ap.add_argument("--background-m", type=float, default=2.0, help="как [geophys.maggr.finds_points]")
    ap.add_argument("--radius-m", type=float, default=2.0, help="подавление пиков, как в сдаче")
    ap.add_argument("--anonymize", choices=("auto", "yes", "no"), default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8")) if args.gt else None
    anon = args.anonymize == "yes" or (args.anonymize == "auto" and gt is None)
    rows = []
    for i, reg in enumerate(scan_root(args.root)):
        for mt in ("maggr", "etgr", "terfl"):
            all_files = reg.rasters.get(mt, [])
            chosen = set(geophys.select_rasters(all_files, mt))
            for path in all_files:
                rec = {"region": f"region_{i:03d}" if anon else reg.name, "markup": mt,
                       "file": f"{mt}_{len(rows):03d}{path.suffix}" if anon else path.name,
                       "used_by_detector": path in chosen, "utm_json": read_utm_json(reg.utm_json)}
                try:
                    rec.update(raster_passport(path))
                    stats, peaks = peaks_at_floors(path, args.band_mode, args.background_m, args.radius_m)
                    rec.update(stats)
                    if gt is not None and mt == "maggr":
                        pts = gt_points(gt, reg.name, mt, peaks[3])
                        rec["gt_finds"] = len(pts)
                        if len(pts):
                            rec["recall_ceiling"] = gt_ceiling(pts, peaks)
                            rec["gt_strength_mad"] = gt_strength(pts, peaks)
                except Exception as exc:  # один битый файл не останавливает паспорт
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(rec)
                short = {k: rec.get(k) for k in ("region", "file", "bands", "dtype", "crs", "channels_differ",
                                                  "top_value_share", "peaks", "gt_finds", "recall_ceiling")}
                print(json.dumps(short, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
