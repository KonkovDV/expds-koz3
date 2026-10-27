#!/usr/bin/env python3
"""Есть ли эталон трейна внутри рамок публичной выборки: память, дыры или вырезанные данные.

Запускать только на настоящих входах выборки. Сводка `notes/site_overlap_public.json`
посчитана по `mock_val`, а эта папка собрана из копий участков трейна, поэтому
«все рамки внутри трейна, расстояние 0 м» там по построению и о выборке ничего не говорит.
Рамка участка трейна — объединение его файлов, и внутри неё могут быть дыры. Скрипт смотрит
каждый файл выборки отдельно и различает три мира:

  A. Файлы трейна того же типа покрывают рамку, эталон трейна внутри не реже, чем вокруг.
     Объекты выборки, вероятно, те же, что в трейне. Модель, выучившая участок, должна брать
     на площадке больше честного балла; у нас этого не случилось (0,32850 при честном 0,329489).
  B. Файлы трейна покрывают рамку, а эталона внутри нет или он заметно реже (ratio < 0,2).
     Разметку вырезали, растры оставили: обучение учит «фон» ровно на объектах выборки.
  C. Файлов трейна того же типа в рамке почти нет (cover < 0,2): вырезаны сами данные.

С `--sample N` растры трейна проверяются в N точках рамки: есть ли там данные, а не nodata.
Так виден вырез внутри файла, который рамкой файла не ловится.

    python scripts/val_memory.py --train F:/2/train-expds-3 --gt F:/expds3/work/gt_train_v2.geojson \
        --val <папка публичной выборки> --out notes/val_memory_public.json --sample 400

Эталон площадки по типам данных из лога 24.10 (`docs/LB_LOG.md`): лидар 964, спутник 377,
аэрофото 75, геофизика 1441. Сумма `gt_same_type` по типу против этих чисел показывает, какая
доля эталона площадки могла бы совпасть с эталоном трейна. Имена участков выборки в вывод не
попадают (номера по порядку).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import box as sbox
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PLATFORM_GT = {"li": 964, "sp_or": 377, "ae": 75, "geophysics": 1441}
GEOPHYSICS = ("maggr", "etgr", "terfl")


def platform_type(markup: str | None) -> str:
    m = (markup or "").lower()
    return "geophysics" if m in GEOPHYSICS else m


def km2(b: tuple) -> float:
    """Площадь рамки в градусах, км² (для рамок до десятков километров хватает)."""
    lat = math.radians((b[1] + b[3]) / 2)
    return (b[2] - b[0]) * 111.32 * math.cos(lat) * (b[3] - b[1]) * 110.574


def geom_km2(g, lat: float) -> float:
    return g.area * 111.32 * math.cos(math.radians(lat)) * 110.574


def assess(vbox: tuple, mtype: str, train: list[tuple], lon: np.ndarray, lat: np.ndarray,
           cls: np.ndarray, mt: np.ndarray) -> dict:
    """Один файл выборки: покрытие файлами трейна того же типа и плотность эталона внутри и вокруг.

    `train` — список (тип, рамка) файлов трейна; точки эталона — представительные точки в EPSG:4326.
    """
    vb = sbox(*vbox)
    same = [sbox(*b) for t, b in train if platform_type(t) == platform_type(mtype) and sbox(*b).intersects(vb)]
    union = unary_union(same) if same else None
    cover = union.intersection(vb).area / vb.area if union is not None and vb.area > 0 else 0.0
    inside = (lon >= vbox[0]) & (lon <= vbox[2]) & (lat >= vbox[1]) & (lat <= vbox[3])
    same_type = np.array([platform_type(m) == platform_type(mtype) for m in mt], dtype=bool) if len(mt) else inside
    rec = {"type": mtype, "area_km2": round(km2(vbox), 4), "train_cover": round(cover, 3),
           "gt_in": {str(c): int(n) for c, n in zip(*np.unique(cls[inside], return_counts=True))},
           "gt_same_type": int((inside & same_type).sum())}
    mid = (vbox[1] + vbox[3]) / 2
    a_in = geom_km2(union.intersection(vb), mid) if union is not None else 0.0
    if union is not None:
        ring = union.difference(vb)
        a_out = geom_km2(ring, mid)
        rb = ring.bounds if not ring.is_empty else None
        n_out = 0
        if rb is not None and a_out > 0:
            from shapely import contains_xy

            cand = same_type & ~inside & (lon >= rb[0]) & (lon <= rb[2]) & (lat >= rb[1]) & (lat <= rb[3])
            n_out = int(contains_xy(ring, lon[cand], lat[cand]).sum()) if cand.any() else 0
        d_in = rec["gt_same_type"] / a_in if a_in > 0 else None
        d_out = n_out / a_out if a_out > 0 else None
        rec.update({"dens_in": None if d_in is None else round(d_in, 3),
                    "dens_out": None if d_out is None else round(d_out, 3),
                    "ratio": round(d_in / d_out, 3) if d_in is not None and d_out else None})
    rec["regime"] = regime(rec)
    return rec


def regime(rec: dict) -> str:
    if rec["train_cover"] < 0.2:
        return "C"
    r = rec.get("ratio")
    if rec["train_cover"] >= 0.8 and r is not None:
        if r >= 0.5:
            return "A"
        if r < 0.2:
            return "B"
    if rec["train_cover"] >= 0.8 and r is None and rec["gt_same_type"] == 0:
        return "B?"
    return "смешанный"


def valid_fraction(vbox: tuple, paths: list[Path], n: int, seed: int = 0) -> float | None:
    """Доля из n случайных точек рамки, где хотя бы один растр трейна несёт данные (не nodata)."""
    import rasterio
    from pyproj import Transformer

    if not paths or n <= 0:
        return None
    rng = np.random.default_rng(seed)
    xs = rng.uniform(vbox[0], vbox[2], n)
    ys = rng.uniform(vbox[1], vbox[3], n)
    hit = np.zeros(n, dtype=bool)
    for p in paths:
        try:
            with rasterio.open(p) as ds:
                if ds.crs is None:
                    continue
                tx, ty = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform(xs, ys)
                b = ds.bounds
                ok = (tx >= b.left) & (tx <= b.right) & (ty >= b.bottom) & (ty <= b.top) & ~hit
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                vals = np.array([v for v in ds.sample(list(zip(tx[idx], ty[idx])), indexes=1)]).ravel()
                good = np.isfinite(vals.astype("float64"))
                if ds.nodata is not None:
                    good &= vals != ds.nodata
                good &= vals != 0
                hit[idx[good]] = True
        except Exception as exc:  # noqa: BLE001 — битый или непривязанный файл трейна просто не голосует
            print(f"растр трейна пропущен: {Path(p).name}: {exc}", file=sys.stderr)
    return round(float(hit.mean()), 3)


def file_boxes(reg) -> list[tuple[str, tuple, Path | None]]:
    """(тип, рамка EPSG:4326, путь растра или None для LAS) по всем привязанным файлам участка."""
    from site_overlap import las_lonlat_box, raster_lonlat_box

    from expds.catalog import read_utm_json

    utm = read_utm_json(reg.utm_json)
    out = []
    for mtype, files in reg.rasters.items():
        for p in files:
            try:
                b = raster_lonlat_box(p)
            except Exception:  # noqa: BLE001
                b = None
            if b and all(map(math.isfinite, b)):
                out.append((mtype, b, p))
    for p in reg.las:
        try:
            b = las_lonlat_box(p, utm)
        except Exception:  # noqa: BLE001
            b = None
        if b and all(map(math.isfinite, b)):
            out.append(("li", b, None))
    return out


def gt_points(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Представительные точки эталона трейна в EPSG:4326, класс и тип данных."""
    from expds.crs import WEB_MERCATOR, to_crs

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lon, lat, cls, mt = [], [], [], []
    for f in data.get("features", []):
        g = f.get("geometry")
        if not g:
            continue
        pt = shape(g).representative_point()
        if abs(pt.x) > 180 or abs(pt.y) > 90:
            pt = to_crs(pt, WEB_MERCATOR, "EPSG:4326")
        props = f.get("properties") or {}
        lon.append(pt.x)
        lat.append(pt.y)
        cls.append(props.get("class_name", "?"))
        mt.append(props.get("markup_type", "?"))
    return np.array(lon), np.array(lat), np.array(cls), np.array(mt)


def summarize(rows: list[dict]) -> dict:
    by_type: dict[str, dict] = {}
    for r in rows:
        t = platform_type(r["type"])
        s = by_type.setdefault(t, {"files": 0, "gt_same_type": 0, "regimes": {}})
        s["files"] += 1
        s["gt_same_type"] += r["gt_same_type"]
        s["regimes"][r["regime"]] = s["regimes"].get(r["regime"], 0) + 1
    for t, s in by_type.items():
        if t in PLATFORM_GT:
            s["platform_gt"] = PLATFORM_GT[t]
            s["share_of_platform_gt"] = round(s["gt_same_type"] / PLATFORM_GT[t], 3)
    return by_type


def main() -> None:
    from expds.catalog import scan_root

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--gt", required=True, help="эталон трейна (GeoJSON, как у табло)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--sample", type=int, default=0, help="точек на рамку для проверки nodata растров трейна")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = []
    for reg in scan_root(args.train):
        train.extend(file_boxes(reg))
    lon, lat, cls, mt = gt_points(args.gt)
    rows = []
    for i, reg in enumerate(scan_root(args.val)):
        for k, (mtype, vbox, _) in enumerate(file_boxes(reg)):
            rec = {"region": f"val_{i:03d}", "file": k,
                   **assess(vbox, mtype, [(t, b) for t, b, _ in train], lon, lat, cls, mt)}
            if args.sample:
                paths = [p for t, b, p in train if p is not None and platform_type(t) == platform_type(mtype)
                         and sbox(*b).intersects(sbox(*vbox))]
                rec["train_valid"] = valid_fraction(vbox, paths, args.sample)
            rows.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"files": rows, "summary": summary}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
