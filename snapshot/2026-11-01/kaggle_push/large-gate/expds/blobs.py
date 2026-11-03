"""Кандидаты курганов на оптике (Ae, SpOr) без видеокарты: пятна «центр — окружение».

Курган на аэрофото и спутнике — светлое или тёмное пятно (по сезону и
распашке). Отклик на масштабе a — среднее круга радиуса a минус среднее
окружения радиуса surround·a; пики модуля отклика по нескольким масштабам —
кандидаты, их признаки — отклики со знаком. Оценка — LightGBM вне обучения
по «семьям» участков (как у кандидатов по рельефу, expds.candidates).

Радиусы 2/4/7 м ловят рябь, а не насыпь: медиана кургана на спутнике около
1000 м² (радиус ~18 м). Для насыпей — `--radii 8,16,28 --peak-radius 12`.

    python -m expds.blobs oof --root <данные> --gt work/gt_train.geojson --markup ae --out work/blobs_ae.geojson
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import accel

log = logging.getLogger(__name__)


@dataclass
class BlobParams:
    work_res: float = 0.5
    radii_m: tuple = (2.0, 4.0, 7.0)
    surround: float = 2.5
    peak_radius_m: float = 4.0
    min_contrast: float = 0.2      # |отклик| в σ снимка
    circle_m: float = 2.0
    block: int = 4096
    max_candidates: int = 100_000


def gray_stats(path: str | Path, max_side: int = 2048) -> tuple[float, float]:
    """Среднее и σ яркости снимка (среднее первых ≤3 каналов) по уменьшенному обзору; 0 — нет данных."""
    import math

    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        n = min(ds.count, 3)
        f = max(1, math.ceil(max(ds.width, ds.height) / max_side))
        a = ds.read(list(range(1, n + 1)), out_shape=(n, max(1, ds.height // f), max(1, ds.width // f)),
                    resampling=Resampling.nearest).astype(np.float32)
    g = a.mean(axis=0)
    valid = (a != 0).any(axis=0)
    return (float(g[valid].mean()), float(g[valid].std()) or 1.0) if valid.any() else (0.0, 1.0)


def detect(path: str | Path, p: BlobParams | None = None) -> dict:
    """Кандидаты снимка: x, y в CRS растра, признаки N×(2·масштабов+1), crs."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    from .tiles import working_res

    p = p or BlobParams()
    mean, sigma = gray_stats(path)
    xs, ys, feats = [], [], []
    with rasterio.open(path) as ds:
        r = working_res(abs(ds.res[0]), p.work_res)
        scale = r / abs(ds.res[0])
        w, h = max(1, int(ds.width / scale)), max(1, int(ds.height / scale))
        n = min(ds.count, 3)
        crs = ds.crs.to_string() if ds.crs else None
        radii = [max(1, round(a / r)) for a in p.radii_m]
        overlap = 2 * round(max(p.radii_m) * p.surround / r)
        half = overlap // 2
        step = max(1, p.block - overlap)
        bx = list(range(0, max(w - overlap, 1), step))
        by = list(range(0, max(h - overlap, 1), step))
        for iy, y0 in enumerate(by):
            for ix, x0 in enumerate(bx):
                bw, bh = min(p.block, w - x0), min(p.block, h - y0)
                win = Window(x0 * scale, y0 * scale, bw * scale, bh * scale)
                a = ds.read(list(range(1, n + 1)), window=win, out_shape=(n, bh, bw),
                            resampling=Resampling.average if scale > 1 else Resampling.bilinear).astype(np.float32)
                valid = (a != 0).any(axis=0)
                if not valid.any():
                    continue
                z = np.where(valid, (a.mean(axis=0) - mean) / sigma, np.nan).astype(np.float32)
                resp = [accel.box_mean(z, k) - accel.box_mean(z, round(k * p.surround)) for k in radii]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    strength = np.nanmax(np.abs(np.stack(resp)), axis=0)
                strength = np.where(valid, np.nan_to_num(strength, nan=0.0), np.nan).astype(np.float32)
                rows, cols, vals = accel.find_peaks(strength, max(1, round(p.peak_radius_m / r)),
                                                    p.min_contrast, p.max_candidates)
                c0x, c0y = (half if ix > 0 else 0), (half if iy > 0 else 0)
                c1x = bw - (half if ix < len(bx) - 1 else 0)
                c1y = bh - (half if iy < len(by) - 1 else 0)
                core = (cols >= c0x) & (cols < c1x) & (rows >= c0y) & (rows < c1y)
                rows, cols, vals = rows[core], cols[core], vals[core]
                if rows.size == 0:
                    continue
                tr = ds.window_transform(win) @ rasterio.Affine.scale(win.width / bw, win.height / bh)
                px, py = tr.c + (cols + 0.5) * tr.a, tr.f + (rows + 0.5) * tr.e
                f = np.column_stack([np.stack([q[rows, cols] for q in resp], 1), vals,
                                     np.stack([np.abs(q[rows, cols]) for q in resp], 1)])
                xs.append(px)
                ys.append(py)
                feats.append(f)
    if not xs:
        return {"x": np.zeros(0), "y": np.zeros(0), "features": np.zeros((0, 2 * len(p.radii_m) + 1), np.float32),
                "crs": None, "res": None}
    x, y, f = np.concatenate(xs), np.concatenate(ys), np.concatenate(feats).astype(np.float32)
    if len(x) > p.max_candidates:
        keep = np.argsort(-f[:, len(p.radii_m)])[: p.max_candidates]
        x, y, f = x[keep], y[keep], f[keep]
    return {"x": x, "y": y, "features": f, "crs": crs, "res": r}


def collect(root: str | Path, gt_path: str | Path, markup: str, p: BlobParams | None = None) -> list[dict]:
    """Кандидаты и метки по всем снимкам модальности: по записи на снимок."""
    from shapely.geometry import shape

    from .candidates import label
    from .catalog import scan_root
    from .naming import site_family

    p = p or BlobParams()
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))["features"]
    by_region: dict[str, list[dict]] = {}
    for f in gt:
        pr = f["properties"]
        if pr["markup_type"] == markup:
            by_region.setdefault(pr["region_name"], []).append(
                {"geom": shape(f["geometry"]), "class_name": pr["class_name"], "markup_type": markup})
    out = []
    for reg in scan_root(root):
        for path in reg.rasters.get(markup, []):
            c = detect(path, p)
            if len(c["x"]) == 0 or c["crs"] is None:
                continue
            y = label(c, c["crs"], by_region.get(reg.name, []), markup=markup)
            out.append({"region": reg.name, "family": site_family(reg.name), "source": Path(path).name,
                        "c": c, "crs": c["crs"], "y": y, "markup": markup})
            log.info("%s %s: кандидатов %d, позитивов %d", reg.name, Path(path).name[-30:], len(y), int((y > 0).sum()))
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Кандидаты курганов на оптике: LightGBM вне обучения по «семьям»")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof")
    o.add_argument("--root", required=True)
    o.add_argument("--gt", required=True)
    o.add_argument("--markup", required=True, choices=["ae", "sp_or", "or"])
    o.add_argument("--out", required=True)
    o.add_argument("--folds", type=int, default=5)
    o.add_argument("--radii", default="", help="радиусы пятна в метрах через запятую; пусто — 2,4,7")
    o.add_argument("--peak-radius", type=float, default=None, help="минимальный шаг между пиками, м")
    o.add_argument("--min-contrast", type=float, default=None)
    o.add_argument("--circle", type=float, default=None, help="радиус круга кандидата, м")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from . import candidates, post, writer

    kw = {}
    if args.radii:
        kw["radii_m"] = tuple(float(x) for x in args.radii.split(",") if x.strip())
    if args.peak_radius is not None:
        kw["peak_radius_m"] = args.peak_radius
    if args.min_contrast is not None:
        kw["min_contrast"] = args.min_contrast
    if args.circle is not None:
        kw["circle_m"] = args.circle
    p = BlobParams(**kw)
    log.info("масштабы %s м, шаг пиков %s м, контраст %s", p.radii_m, p.peak_radius_m, p.min_contrast)
    recs = collect(args.root, args.gt, args.markup, p)
    cp = candidates.CandidateParams(circle_m=p.circle_m)
    feats = candidates.oof_predictions(recs, cp, args.folds, multiclass=True)
    feats = post.finalize(post.merge_within_markup(feats))
    writer.write_geojson(feats, args.out)
    print(json.dumps({"rasters": len(recs), "families": len({r["family"] for r in recs}),
                      "candidates": int(sum(len(r["y"]) for r in recs)),
                      "positives": int(sum((r["y"] > 0).sum() for r in recs)), "written": len(feats)},
                     ensure_ascii=False))


if __name__ == "__main__":
    _cli()
