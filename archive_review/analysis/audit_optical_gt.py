"""Сдвиг разметки курганов между слоями и форма эталонных полигонов.

Без видеокарты. Если внутри участка сдвиг устойчивый, сеть на снимке учит
не ту точку. Если направления случайные, промах — в локализации.

    python analysis/audit_optical_gt.py --gt F:/expds3/work/gt_train.geojson --pred F:/expds3/work/spor_center_oof.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


KURGANS = {"kurgany_povrezhdennye", "kurgany_tselye"}


def utm_xy(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """EPSG:3857 → метры UTM зоны первой точки."""
    from pyproj import Transformer

    from expds.crs import utm_for_lonlat, WEB_MERCATOR

    lon, lat = Transformer.from_crs(WEB_MERCATOR, "EPSG:4326", always_xy=True).transform(float(xs[0]), float(ys[0]))
    zone = utm_for_lonlat(lon, lat)
    x, y = Transformer.from_crs(WEB_MERCATOR, zone, always_xy=True).transform(xs, ys)
    return np.column_stack([x, y])


def load(path: str, markup: set[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """region → markup → точки N×2 в метрах UTM."""
    fc = json.loads(open(path, encoding="utf-8").read())
    raw: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in fc["features"]:
        p = f.get("properties") or {}
        if p.get("class_name") not in KURGANS:
            continue
        mt = p.get("markup_type") or ""
        if markup is not None and mt not in markup:
            continue
        g = shape(f["geometry"])
        if g.is_empty:
            continue
        c = g.representative_point()
        raw[p.get("region_name") or ""][mt].append((c.x, c.y, g.area, g.length))
    out = {}
    for region, layers in raw.items():
        out[region] = {}
        for mt, rows in layers.items():
            arr = np.asarray(rows, np.float64)
            pts = utm_xy(arr[:, 0], arr[:, 1])
            out[region][mt] = np.column_stack([pts, arr[:, 2], arr[:, 3]])
    return out


def shifts(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """Сдвиги b − a для пар ближе radius метров. Пусто, если пар нет."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2))
    dist, idx = cKDTree(b[:, :2]).query(a[:, :2], distance_upper_bound=radius)
    ok = np.isfinite(dist) & (idx < len(b))
    if not ok.any():
        return np.zeros((0, 2))
    return b[idx[ok], :2] - a[ok, :2]


def circularity(area: np.ndarray, length: np.ndarray) -> np.ndarray:
    return np.where(length > 0, 4 * np.pi * area / length ** 2, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", default="")
    ap.add_argument("--radius", type=float, default=40.0)
    args = ap.parse_args()
    gt = load(args.gt)
    print(f"участков с курганами: {len(gt)}", flush=True)
    pairs = (("li", "ae"), ("li", "sp_or"), ("ae", "sp_or"))
    print("\nсдвиг между слоями эталона, медиана по участку, м", flush=True)
    for a, b in pairs:
        meds = []
        systematic = 0
        n_reg = 0
        for layers in gt.values():
            if a not in layers or b not in layers:
                continue
            d = shifts(layers[a], layers[b], args.radius)
            if len(d) < 5:
                continue
            n_reg += 1
            med = np.median(d, axis=0)
            mad = np.median(np.abs(d - med), axis=0)
            meds.append(med)
            if float(np.hypot(*med)) > 3 and float(np.hypot(*mad)) < 5:
                systematic += 1
        if not meds:
            print(f"  {a} → {b}: пар недостаточно", flush=True)
            continue
        meds = np.asarray(meds)
        print(f"  {a} → {b}: участков {n_reg}, медиана сдвига {np.median(meds, axis=0).round(2)}, "
              f"участков со устойчивым сдвигом > 3 м: {systematic}", flush=True)

    print("\nформа эталона", flush=True)
    for mt in ("sp_or", "ae", "li"):
        areas, circ = [], []
        for layers in gt.values():
            if mt not in layers:
                continue
            row = layers[mt]
            areas.append(row[:, 2])
            circ.append(circularity(row[:, 2], row[:, 3]))
        if not areas:
            continue
        area = np.concatenate(areas)
        c = np.concatenate(circ)
        print(f"  {mt}: n {len(area)}, площадь p50 {np.median(area):.0f} м², "
              f"округлость < 0,5: {(c < 0.5).mean():.2f}, медиана округлости {np.median(c):.2f}", flush=True)

    if not args.pred:
        return
    pred = load(args.pred, markup={"sp_or"})
    print("\nсдвиг предсказания к эталону sp_or, ближе 20 м", flush=True)
    meds = []
    systematic = 0
    n_reg = 0
    for region, layers in gt.items():
        if "sp_or" not in layers or region not in pred or "sp_or" not in pred[region]:
            continue
        d = shifts(layers["sp_or"], pred[region]["sp_or"], 20.0)
        if len(d) < 5:
            continue
        n_reg += 1
        med = np.median(d, axis=0)
        mad = np.median(np.abs(d - med), axis=0)
        meds.append(np.hypot(*med))
        if float(np.hypot(*med)) > 3 and float(np.hypot(*mad)) < 5:
            systematic += 1
    if meds:
        print(f"  участков {n_reg}, медиана |сдвига| {np.median(meds):.1f} м, "
              f"устойчивых сдвигов: {systematic}", flush=True)


if __name__ == "__main__":
    main()
