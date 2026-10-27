#!/usr/bin/env python3
"""Подбор параметров классики геофизики официальной метрикой с проверкой по «семьям».

Для каждой пары (модальность, класс) перебираются параметры детектора; порог
выдачи — по кривой точного сопоставления (expds.metric). Честная оценка — по
«семьям»: параметры и порог выбираются на остальных семьях и проверяются на
отложенной (TP/FP/FN отложенных семей суммируются, как в метрике).

    python scripts/geo_tune.py --root D:/expds3 --gt work/gt_train.geojson --out work/geo_tune.json

Опыты 25.10 (каждый в свой JSON, сравнение — по f1_lofo_pooled, не по f1_in_sample):

    ... --only maggr/finds_points --k-min 1.0 --out work/geo_k10.json            # база сдачи
    ... --only maggr/finds_points --k-min 0.5 0.75 1.0 --out work/geo_k.json     # отсечка ниже 1 MAD
    ... --only maggr/finds_points --band-mode auto --out work/geo_band.json      # RGB-карты через PCA
    ... --only maggr/finds_points --signals abs asa --out work/geo_asa.json      # аналитический сигнал
    ... --only maggr/object_poly --polarities abs pos --out work/geo_pol.json    # ямы — положительные
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import geophys, metric, post
from expds.catalog import read_utm_json, scan_root
from expds.naming import site_family
from expds.raster import circle_3857, pixel_to_xy
from expds.writer import to_feature_collection

log = logging.getLogger("geo_tune")

#: Параметры детектора (пересчитываются) и фильтры после него (use_noise).
GRID = {
    ("maggr", "finds_points"): {"background_m": [1.0, 2.0, 4.0, 8.0], "peak_radius_m": [0.5, 1.0, 2.0],
                                "peak_k": [0.5, 0.75, 1.0]},
    ("maggr", "object_poly"): {"background_m": [4.0, 8.0, 16.0], "smooth_m": [0.5, 1.0, 2.0],
                               "poly_k": [1.0, 1.5, 2.0, 3.0]},
    ("etgr", "object_poly"): {"background_m": [1.0, 2.0, 4.0], "smooth_m": [0.1, 0.25, 0.5],
                              "poly_k": [1.0, 1.5, 2.0, 3.0]},
    ("terfl", "object_poly"): {"background_m": [2.0, 4.0, 8.0], "smooth_m": [0.25, 0.5, 1.0],
                               "poly_k": [1.0, 1.5, 2.0, 3.0]},
}
NOISE_OPTIONS = {"maggr": [False, True]}
CONFIDENCE_OPTIONS = ["strength", "rank"]
#: Нижняя отсечка пиков (MAD) по умолчанию: у эталонных точек Семибратнего медиана силы 1,9.
#: До 24.10 она не перебиралась. На публичной выборке геофизика выдала 570 объектов на 1441
#: эталон при 65 % попаданий, а добавлять выгодно уже от 11 %: полноту держит эта отсечка,
#: а не порог ранга. Сетка peak_k у finds_points проверяет 0,5 и 0,75.
K_MIN = 1.0


def detect_all(data: dict, mt: str, cls: str, grid: dict, band_mode: str = "first") -> dict:
    """{конфиг детектора: {регион: объекты}}; фильтр считается один раз на (растр, фон).

    Пики ищутся один раз на (фон, радиус) с самой низкой отсечкой сетки и затем
    режутся по peak_k: жадное подавление идёт по убыванию силы, поэтому набор
    пиков выше k тот же, что при поиске сразу с отсечкой k.
    """
    keys = list(grid)
    configs = [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]
    out = {json.dumps(c, sort_keys=True): defaultdict(list) for c in configs}
    for region, items in data.items():
        for path in items["rasters"]:
            a, tr, crs, res = geophys.read_band(path, band_mode)
            z = geophys.robust_z(a)
            del a
            for bg in sorted({c["background_m"] for c in configs}):
                hp = geophys.highpass(z, res, bg)
                peaks: dict[tuple, tuple] = {}
                images: dict[str, np.ndarray] = {}
                for c in (c for c in configs if c["background_m"] == bg):
                    p = geophys.GeoParams(**c)
                    p.min_area_m2, p.max_area_m2 = geophys.AREA_M2[mt]
                    if cls == "finds_points":
                        floor = min(grid.get("peak_k", [K_MIN]))
                        key = (p.peak_radius_m, p.signal)
                        if key not in peaks:
                            if p.signal not in images:
                                images[p.signal] = geophys.detection_image(hp, res, p.signal)
                            rows, cols, vals = geophys.accel.find_peaks(
                                images[p.signal], max(1, round(p.peak_radius_m / res)), floor)
                            xs, ys = pixel_to_xy(tr, rows, cols)
                            peaks[key] = (xs, ys, vals)
                        xs, ys, vals = peaks[key]
                        keep = vals >= np.float32(c.get("peak_k", K_MIN))
                        out[json.dumps(c, sort_keys=True)][region].append(
                            (xs[keep], ys[keep], vals[keep], crs, path.name))
                    else:
                        out[json.dumps(c, sort_keys=True)][region] += geophys.polys_from(
                            hp, tr, crs, res, region, mt, path.name, p)
                del hp, images
    return out


def features_of(region: str, mt: str, items) -> list[dict]:
    """Объекты региона из сохранённых результатов детектора (точки хранятся массивами)."""
    feats = []
    for it in items:
        if isinstance(it, dict):
            feats.append(it)
            continue
        xs, ys, vals, crs, src = it
        feats += [{"geom": circle_3857(float(x), float(y), 0.3, crs), "class_name": "finds_points",
                   "confidence": float(geophys.strength_confidence(v)), "region_name": region,
                   "markup_type": mt, "source": src, "original_crs": crs} for x, y, v in zip(xs, ys, vals)]
    return feats


def counts(curve, regions: set[str], tau: float) -> tuple[int, int, int]:
    keep = (curve.conf >= tau) & np.array([r in regions for r in curve.pred_region], bool)
    tp = int(curve.hit[keep].sum())
    n_gt = sum(v for r, v in curve.gt_regions.items() if r in regions)
    return tp, int(keep.sum()) - tp, n_gt - tp


def f1(tp, fp, fn) -> float:
    return float(metric._f1(tp, fp, fn))


def best_tau(curve, regions: set[str]) -> tuple[float, float]:
    """Лучший F1 и порог на подмножестве регионов (перебор всех различимых порогов)."""
    sel = np.array([r in regions for r in curve.pred_region], bool)
    n_gt = sum(v for r, v in curve.gt_regions.items() if r in regions)
    conf, hit = curve.conf[sel], curve.hit[sel]
    if conf.size == 0 or n_gt == 0:
        return 0.0, float("inf")
    last = np.r_[np.flatnonzero(conf[1:] != conf[:-1]), conf.size - 1]
    tp = np.cumsum(hit)[last].astype(float)
    n = (last + 1).astype(float)
    scores = metric._f1(tp, n - tp, n_gt - tp)
    k = int(np.argmax(scores))
    return float(scores[k]), float(conf[last[k]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="например maggr/finds_points")
    ap.add_argument("--band-mode", default="first", choices=geophys.BAND_MODES,
                    help="как RGB-карта сводится к числу; сданный балл снят с first")
    ap.add_argument("--k-min", type=float, nargs="*", default=None,
                    help="сетка нижней отсечки пиков finds_points, MAD (по умолчанию 0,5 0,75 1,0)")
    ap.add_argument("--signals", nargs="*", default=None, choices=geophys.SIGNALS,
                    help="finds_points: abs (сдача) и/или asa — аналитический сигнал")
    ap.add_argument("--polarities", nargs="*", default=None, choices=geophys.POLARITIES,
                    help="object_poly: abs (сдача), pos, neg")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    regions = scan_root(args.root)
    out_path = Path(args.out)
    result = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for (mt, cls), grid in GRID.items():
        if args.only and args.only != f"{mt}/{cls}":
            continue
        data = {}
        for reg in regions:
            files = geophys.select_rasters(reg.rasters.get(mt, []), mt)
            if not files:
                continue
            utm = read_utm_json(reg.utm_json)
            data[reg.name] = {"rasters": files, "family": site_family(reg.name),
                              "boundary": geophys.survey_boundary(reg.boundaries, mt, utm, {}),
                              "noise": geophys.load_mask(reg.noise, mt, utm)}
        g = {**gt, "features": [f for f in gt["features"] if f["properties"]["markup_type"] == mt
                                and f["properties"]["class_name"] == cls and f["properties"]["region_name"] in data]}
        if not data or not g["features"]:
            continue
        fams = defaultdict(set)
        for r, v in data.items():
            fams[v["family"]].add(r)
        if cls == "finds_points" and args.k_min:
            grid = {**grid, "peak_k": sorted(args.k_min)}
        if cls == "finds_points" and args.signals:
            grid = {**grid, "signal": list(args.signals)}
        if cls == "object_poly" and args.polarities:
            grid = {**grid, "polarity": list(args.polarities)}
        det = detect_all(data, mt, cls, grid, args.band_mode)
        curves = []
        for key, by_region in det.items():
            for use_noise, conf in itertools.product(NOISE_OPTIONS.get(mt, [False]), CONFIDENCE_OPTIONS):
                feats = []
                for region, items in data.items():
                    found = features_of(region, mt, by_region.get(region, []))
                    if conf == "rank":
                        found = geophys.rank_within_source(found)
                    feats += geophys.apply_masks(found, items["boundary"], items["noise"] if use_noise else None)
                pred = to_feature_collection(post.finalize(post.merge_within_markup(feats)))
                cfg = {**json.loads(key), "use_noise": use_noise, "confidence": conf, "band_mode": args.band_mode}
                if cls == "finds_points":
                    cfg.setdefault("peak_k", K_MIN)
                curves.append((cfg, metric.build_curves(pred, g, args.metric)[cls]))
        allr = set(data)
        (s_all, t_all), cfg_all = max(((best_tau(c, allr), cfg) for cfg, c in curves), key=lambda t: t[0][0])
        tp = fp = fn = 0
        lofo = []
        for fam, held in sorted(fams.items()):
            train = allr - held
            if not train:
                continue
            (_, t), cfg, c = max(((best_tau(c, train), cfg, c) for cfg, c in curves), key=lambda x: x[0][0])
            a, b, d = counts(c, held, t)
            tp, fp, fn = tp + a, fp + b, fn + d
            lofo.append({"held_out": fam, "cfg": cfg, "tau": t, "f1_held": round(f1(a, b, d), 4),
                         "tp": a, "fp": b, "fn": d})
        rec = {"best_cfg": cfg_all, "tau": t_all, "f1_in_sample": round(s_all, 4),
               "f1_lofo_pooled": round(f1(tp, fp, fn), 4) if lofo else None,
               "n_gt": len(g["features"]), "regions": len(data), "families": len(fams), "lofo": lofo}
        result[f"{mt}/{cls}"] = rec
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{mt:6s} {cls:13s} GT {rec['n_gt']:5d}  регионов {rec['regions']:2d}  семей {rec['families']}  "
              f"F1 на всех {s_all:.3f}  F1 по семьям {rec['f1_lofo_pooled']}  лучшее {cfg_all}, порог {t_all:.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
