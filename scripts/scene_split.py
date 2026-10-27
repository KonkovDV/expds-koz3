#!/usr/bin/env python3
"""Две группы порога курганов спутника: большая сцена и остальные.

Площадь — на земле, в км², по заголовку снимка, не по размеру файла.
Порог — середина самого большого относительного разрыва среди крупных сцен.
15 самых больших участков обязаны оказаться по одну сторону: разрыв ищется
не раньше 6-го места. Имена участков в отчёт не пишутся.

Дальше одно табло по семьям:

    python scripts/scene_split.py --root F:/2/train-expds-3 \
        --src F:/expds3/work/spor_focal_oof.geojson \
        --out F:/expds3/work/spor_focal_scene.geojson
    python scripts/scoreboard.py <флаги раздела 0> --replace sp_focal=<out> --honest 10

Если порог группы sp_focal_big совпадёт с порогом sp_focal до 0,01 — разницы нет.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BIG = "sp_focal_big"
BASE = "sp_focal"


def scene_cut(areas: list[float], start: int = 5, window: int = 40) -> float:
    """Середина наибольшего относительного разрыва, не раньше ``start``-го места.

    ``start`` — индекс в отсортированном по убыванию списке. Разрыв между
    первыми одиночными сценами не берётся: они и так по одну сторону с остальными
    крупными.
    """
    xs = sorted((float(a) for a in areas if a and a > 0), reverse=True)
    if len(xs) < start + 2:
        raise ValueError("слишком мало сцен, чтобы провести границу")
    last = min(window, len(xs) - 1)
    best_i, best = start, -1.0
    for i in range(start, last):
        rel = (xs[i] - xs[i + 1]) / xs[i + 1]
        if rel > best:
            best, best_i = rel, i
    return (xs[best_i] + xs[best_i + 1]) / 2.0


def ground_km2(path: Path) -> float | None:
    """Площадь растра на земле, км². Читается только заголовок."""
    import rasterio

    from expds.raster import raster_kind_and_latitude

    try:
        with rasterio.open(path) as ds:
            kind, lat = raster_kind_and_latitude(ds.crs, ds.transform, ds.width, ds.height)
            rx, ry = abs(float(ds.res[0])), abs(float(ds.res[1]))
            cos = max(math.cos(math.radians(lat)), 0.2)
            if kind == "meter":
                pixel = rx * ry
            elif kind == "3857":
                pixel = rx * ry * cos * cos
            else:
                pixel = rx * ry * (111_320.0 * cos) ** 2
            return ds.width * ds.height * pixel / 1e6
    except Exception:
        return None


def region_areas(root: Path) -> dict[str, float]:
    """Крупнейший снимок спутника участка, км² на земле."""
    from expds.catalog import scan_root

    out: dict[str, float] = {}
    for reg in scan_root(root):
        best = 0.0
        for path in reg.rasters.get("sp_or") or []:
            area = ground_km2(path)
            if area:
                best = max(best, area)
        if best > 0:
            out[reg.name] = best
    return out


def assign(features: list[dict], big_regions: set[str]) -> dict:
    """Ставит sp_focal_big на больших сценах и sp_focal на остальных. Считает без имён."""
    n_big = n_base = 0
    regions_big: set[str] = set()
    regions_base: set[str] = set()
    for feat in features:
        props = feat.setdefault("properties", {})
        region = str(props.get("region_name") or "")
        if region in big_regions:
            props["threshold_group"] = BIG
            n_big += 1
            regions_big.add(region)
        else:
            props["threshold_group"] = BASE
            n_base += 1
            regions_base.add(region)
    return {"features_big": n_big, "features_base": n_base,
            "regions_big": len(regions_big), "regions_base": len(regions_base)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--src", required=True, help="карта центров спутника вне обучения, группа ещё не стоит")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    areas = region_areas(Path(args.root))
    cut = scene_cut(list(areas.values()))
    big = {name for name, area in areas.items() if area >= cut}
    fc = json.loads(Path(args.src).read_text(encoding="utf-8"))
    counts = assign(fc.get("features", []), big)
    Path(args.out).write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    ranked = sorted(areas.values(), reverse=True)
    report = {"cut_km2": round(cut, 4), "scenes": len(areas),
              "largest_km2": [round(a, 3) for a in ranked[:20]], **counts}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
