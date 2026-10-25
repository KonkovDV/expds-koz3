#!/usr/bin/env python3
"""Где теряются объекты геофизики: счёт после каждого шага сдачи на каждой карте.

Тот же путь, что `pipeline.geophys_feats`: выбор карт → детектор → ранг внутри карты →
маска границ → маска шума → порог модальности. Для каждой карты печатает число объектов
после каждого шага, привязку растра (CRS, тождественная матрица), рамку в градусах и долю
рамки карты под маской границ и под маской шума.

Зачем. По арифметике лога (`scripts/geo_regime.py`) один из двух классов геофизики на
публичной выборке дал почти ноль попаданий, хотя вне обучения по семьям находки дают
F1 0,46, object_poly — 0,34. Если на карте «после детектора» сотни объектов, а
«после маски» ноль — виновата маска (не та CRS, перевёрнутый полигон, шум на всю
карту). Если ноль уже после детектора — чтение растра (каналы, nodata, привязка).
Если объекты есть до конца, но лежат вне рамки участка — привязка.

    python scripts/geo_stages.py --root <папка публичной выборки> --out notes/geo_stages_public.json
    python scripts/geo_stages.py --root F:/expds3/train-expds-3 --names --out notes/geo_stages_train.json

Имена участков и файлов публичной выборки по умолчанию заменяются номерами.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import geophys
from expds.catalog import scan_root
from expds.pipeline import load_config, region_crs


def footprint_3857(path: Path, fallback_crs: str | None = None):
    """Рамка растра в EPSG:3857 (shapely) или None; плюс объявленная CRS, взятая CRS и тождественная матрица."""
    import rasterio
    from rasterio.warp import transform_bounds
    from shapely.geometry import box

    from expds.crs import to_3857_ok, usable_raster_crs

    with rasterio.open(path) as ds:
        declared = ds.crs.to_string() if ds.crs else None
        crs = usable_raster_crs(declared, ds.bounds, fallback_crs)
        info = {"crs": (declared or None) and declared[:60], "crs_used": crs and crs[:60],
                "crs_declared_usable": to_3857_ok(declared),
                "identity_transform": bool(ds.transform.is_identity), "bands": ds.count,
                "dtype": ds.dtypes[0], "size": [ds.width, ds.height], "res": round(float(abs(ds.res[0])), 4)}
        if not to_3857_ok(crs):
            return None, info
        b = transform_bounds(crs, "EPSG:3857", *ds.bounds, densify_pts=5)
        lon = transform_bounds(crs, "EPSG:4326", *ds.bounds, densify_pts=5)
        info["lonlat_box"] = [round(v, 5) for v in lon]
        return box(*b), info


def cover_share(fp, mask) -> float | None:
    """Доля рамки карты под маской (0…1) или None, если чего-то нет."""
    if fp is None or mask is None or fp.area <= 0:
        return None
    return round(float(fp.intersection(mask).area / fp.area), 4)


def stages_for(path: Path, reg, mt: str, cls: str, p, o, boundary, noise) -> dict:
    """Числа объектов после каждого шага для одной карты и класса."""
    fallback = region_crs(reg, {})
    found = (geophys.finds_points(path, reg.name, mt, p, fallback_crs=fallback) if cls == "finds_points"
             else geophys.object_polys(path, reg.name, mt, p, fallback_crs=fallback))
    rec = {"detector": len(found)}
    if o["rank"]:
        found = geophys.rank_within_source(found)
    in_b = geophys.apply_masks(found, boundary, None)
    rec["after_boundary"] = len(in_b)
    in_n = geophys.apply_masks(in_b, None, noise if o["use_noise"] else None)
    rec["after_noise"] = len(in_n)
    kept = [f for f in in_n if f["confidence"] >= o["min_confidence"]]
    rec["after_min_confidence"] = len(kept)
    if kept:
        from shapely.geometry import MultiPoint
        c = MultiPoint([f["geom"].representative_point() for f in kept]).envelope.centroid
        rec["kept_center_3857"] = [round(c.x, 1), round(c.y, 1)]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    ap.add_argument("--names", action="store_true", help="печатать настоящие имена участков и файлов (трейн)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    geo_cfg = cfg.get("geophys", {})
    rows = []
    for i, reg in enumerate(scan_root(args.root)):
        crs_region = region_crs(reg, cfg)
        raster_skipped = [f for f in reg.skipped if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")]
        if raster_skipped:
            # Неузнанное имя растра: карта MagGr без кода и без слова «магнит» сюда и попадёт
            rec = {"region": reg.name if args.names else f"region_{i:03d}", "skipped_rasters": len(raster_skipped),
                   "files": [f.name if args.names else f"skipped_{k:03d}{f.suffix}" for k, f in enumerate(raster_skipped)]}
            rows.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        for mt in ("maggr", "etgr", "terfl"):
            all_files = reg.rasters.get(mt, [])
            if not all_files:
                continue
            files = geophys.select_rasters(all_files, mt)
            classes = ("finds_points", "object_poly") if mt == "maggr" else ("object_poly",)
            boundary = geophys.load_mask(reg.boundaries, mt, crs_region) if mt != "etgr" else None
            noise = geophys.load_mask(reg.noise, mt, crs_region)
            for j, path in enumerate(all_files):
                rec = {"region": reg.name if args.names else f"region_{i:03d}", "markup": mt,
                       "file": path.name if args.names else f"{mt}_{j:03d}{path.suffix}",
                       "used_by_detector": path in files, "region_crs": crs_region,
                       "boundary_files": len(reg.boundaries), "noise_files": len(reg.noise)}
                try:
                    fp, info = footprint_3857(path, crs_region)
                    rec.update(info)
                    rec["boundary_cover"] = cover_share(fp, boundary)
                    rec["noise_cover"] = cover_share(fp, noise)
                    if path in files:
                        b_map, n_map = geophys.sane_masks(path, boundary, noise, geo_cfg, crs_region)
                        rec["masks_used"] = {"boundary": b_map is not None, "noise": n_map is not None}
                        for cls in classes:
                            p, o = geophys.params_for(geo_cfg, mt, cls)
                            rec[cls] = stages_for(path, reg, mt, cls, p, o, b_map, n_map)
                            if fp is not None and rec[cls].get("kept_center_3857"):
                                from shapely.geometry import Point
                                rec[cls]["kept_center_in_map"] = bool(fp.buffer(50).contains(
                                    Point(*rec[cls]["kept_center_3857"])))
                            rec[cls].pop("kept_center_3857", None)
                except Exception as exc:  # одна битая карта не останавливает разбор
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
