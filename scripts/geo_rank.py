#!/usr/bin/env python3
"""Опыт: переоценка пиков MagGr (finds_points) LightGBM по форме аномалии — вне обучения по «семьям».

Ранг силы пика внутри карты (geophys.rank_within_source) не видит формы: диполь,
одиночный острый пик и край шумной полосы получают одну уверенность. Здесь у
каждого пика — отклики на нескольких масштабах, кольцо вокруг и плотность
соседних пиков; оценщик учится на двух «семьях» и проверяется на третьей.
Результат (13 карт, 3 «семьи»): на тех же пиках ранг 0,582, LightGBM 0,568;
--pre-mask (вне границ съёмки — NaN до нормировки) — ранг 0,572. В решение не вошло.

    python scripts/geo_rank.py oof --root F:/expds3/eval_geo --gt work/gt_train.geojson --out work/geo_rank.geojson
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import accel, geophys  # noqa: E402

log = logging.getLogger(__name__)


def inside_mask(geom, shape: tuple[int, int], transform, crs: str | None) -> np.ndarray:
    """Пиксели растра внутри полигона EPSG:3857 (центр пикселя в полигоне)."""
    from rasterio.features import geometry_mask

    from expds.crs import WEB_MERCATOR, to_crs

    g = to_crs(geom, WEB_MERCATOR, crs) if crs and crs != WEB_MERCATOR else geom
    return geometry_mask([g], out_shape=shape, transform=transform, invert=True)

FEATURE_NAMES = ["v", "sign", "rank", "hp1", "hp2", "hp4", "hp8", "disk", "ring", "ring_std", "prom",
                 "ring_min", "ring_max", "n3", "n8", "level", "res"]


def peak_table(path: str | Path, p: geophys.GeoParams, boundary=None) -> dict:
    """Пики карты (как geophys.finds_points) и их признаки: x, y в CRS растра, features N×17."""
    a, tr, crs, res = geophys.read_band(path)
    if boundary is not None:
        a = np.where(inside_mask(boundary, a.shape, tr, crs), a, np.nan).astype(np.float32)
    z = geophys.robust_z(a)
    hp = geophys.highpass(z, res, p.background_m)
    mag = np.where(np.isfinite(hp), np.abs(hp), np.nan).astype(np.float32)
    rows, cols, v = accel.find_peaks(mag, max(1, round(p.peak_radius_m / res)), p.peak_k)
    if rows.size == 0:
        return {"x": np.zeros(0), "y": np.zeros(0), "features": np.zeros((0, len(FEATURE_NAMES)), np.float32),
                "crs": crs, "res": res}
    rank = np.searchsorted(np.sort(v), v, side="right") / len(v)
    sign = np.sign(hp[rows, cols])
    scales = [geophys.highpass(z, res, b)[rows, cols] * sign for b in (1.0, 2.0, 4.0, 8.0)]
    ring = accel.ring_stats(hp, rows, cols, max(1.0, 1.0 / res), max(2.0, 3.0 / res))
    peaks = np.zeros(hp.shape, np.float32)
    peaks[rows, cols] = 1.0
    dens = []
    for r_m in (3.0, 8.0):
        k = max(1, round(r_m / res))
        dens.append(accel.box_mean(peaks, k)[rows, cols] * (2 * k + 1) ** 2)
    f = np.column_stack([v, sign, rank, *scales, ring, *dens, z[rows, cols], np.full(len(v), res)])
    xs, ys = geophys.pixel_to_xy(tr, rows, cols)
    return {"x": xs, "y": ys, "features": f.astype(np.float32), "crs": crs, "res": res}


def keep_in_masks(c: dict, boundary, noise) -> np.ndarray:
    """Пики внутри границ съёмки и вне шума (маски — EPSG:3857, как geophys.apply_masks)."""
    import shapely

    from expds.crs import WEB_MERCATOR, transformer

    ok = np.ones(len(c["x"]), bool)
    if len(c["x"]) == 0 or (boundary is None and noise is None):
        return ok
    xs, ys = (c["x"], c["y"]) if c["crs"] == WEB_MERCATOR else transformer(c["crs"]).transform(c["x"], c["y"])
    if boundary is not None:
        ok &= shapely.contains_xy(boundary, xs, ys)
    if noise is not None:
        ok &= ~shapely.contains_xy(noise, xs, ys)
    return ok


def collect(root: str | Path, gt_path: str | Path, cfg: dict, pre_mask: bool = False) -> list[dict]:
    """Пики и метки по всем картам MagGr трейна: по записи на карту."""
    from shapely.geometry import shape

    from expds.candidates import label
    from expds.catalog import read_utm_json, scan_root
    from expds.naming import site_family

    p, o = geophys.params_for(cfg.get("geophys", {}), "maggr", "finds_points")
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))["features"]
    by_region: dict[str, list[dict]] = {}
    for f in gt:
        pr = f["properties"]
        if pr["markup_type"] == "maggr" and pr["class_name"] == "finds_points":
            by_region.setdefault(pr["region_name"], []).append(
                {"geom": shape(f["geometry"]), "class_name": "finds_points", "markup_type": "maggr"})
    out = []
    for reg in scan_root(root):
        files = geophys.select_rasters(reg.rasters.get("maggr", []), "maggr")
        if not files:
            continue
        crs_region = read_utm_json(reg.utm_json)
        boundary = geophys.load_mask(reg.boundaries, "maggr", crs_region)
        noise = geophys.load_mask(reg.noise, "maggr", crs_region) if o["use_noise"] else None
        for path in files:
            c = peak_table(path, p, boundary if pre_mask else None)
            if len(c["x"]) == 0 or c["crs"] is None:
                continue
            keep = keep_in_masks(c, boundary, noise)
            c = {**c, "x": c["x"][keep], "y": c["y"][keep], "features": c["features"][keep]}
            y = label(c, c["crs"], by_region.get(reg.name, []), classes=("finds_points",), markup="maggr")
            out.append({"region": reg.name, "family": site_family(reg.name), "source": path.name, "c": c,
                        "crs": c["crs"], "y": y, "point_m": p.point_circle_m})
            log.info("%s %s: пиков %d, у эталона %d", reg.name, path.name[-30:], len(y), int(y.sum()))
    return out


def to_features(r: dict, conf: np.ndarray) -> list[dict]:
    from expds.raster import circle_3857

    return [{"geom": circle_3857(float(x), float(y), r["point_m"], r["crs"]), "class_name": "finds_points",
             "confidence": float(s), "region_name": r["region"], "markup_type": "maggr", "source": r["source"],
             "original_crs": r["crs"]} for x, y, s in zip(r["c"]["x"], r["c"]["y"], conf)]


def oof(recs: list[dict], rounds: int = 300) -> tuple[list[dict], list[dict]]:
    """Оставить одну «семью»: LightGBM на остальных. Возвращает (OOF, базовый ранг) на тех же пиках."""
    import lightgbm as lgb

    from expds.candidates import _lgb_params

    fams = sorted({r["family"] for r in recs})
    feats, base = [], []
    for fam in fams:
        tr = [r for r in recs if r["family"] != fam]
        te = [r for r in recs if r["family"] == fam]
        x = np.nan_to_num(np.vstack([r["c"]["features"] for r in tr]), nan=-999.0)
        y = np.concatenate([r["y"] for r in tr])
        booster = lgb.train(_lgb_params(False), lgb.Dataset(x, y, feature_name=FEATURE_NAMES), num_boost_round=rounds)
        for r in te:
            s = booster.predict(np.nan_to_num(r["c"]["features"], nan=-999.0))
            feats += to_features(r, s)
            base += to_features(r, r["c"]["features"][:, FEATURE_NAMES.index("rank")])
        log.info("семья %s: обучение %d пиков (%d у эталона), проверка %d карт", fam, len(y), int(y.sum()), len(te))
    return feats, base


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Переоценка пиков MagGr LightGBM вне обучения по «семьям»")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof")
    o.add_argument("--root", required=True)
    o.add_argument("--gt", required=True)
    o.add_argument("--out", required=True, help="OOF; рядом *_rank.geojson — базовый ранг на тех же пиках")
    o.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.toml"))
    o.add_argument("--pre-mask", action="store_true", help="вне границ съёмки — NaN до нормировки")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from expds import writer
    from expds.pipeline import load_config

    recs = collect(args.root, args.gt, load_config(args.config), args.pre_mask)
    feats, base = oof(recs)
    out = Path(args.out)
    writer.write_geojson(feats, out)
    writer.write_geojson(base, out.with_name(out.stem + "_rank.geojson"))
    print(json.dumps({"maps": len(recs), "families": len({r["family"] for r in recs}),
                      "peaks": int(sum(len(r["y"]) for r in recs)), "near_gt": int(sum(r["y"].sum() for r in recs))},
                     ensure_ascii=False))


if __name__ == "__main__":
    _cli()
