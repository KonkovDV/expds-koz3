#!/usr/bin/env python3
"""Проверка двух гипотез аудита 29.10 о картах геофизики, как их читает PIXGBM (expds.geo_pixgbm.load_map).

H-G1. Ядро сильной аномалии на визуализации упирается в край шкалы, и load_map считает его «нет данных».
      Сколько эталонных находок (и полюсов object_poly) лежит в таких ядрах? Если заметная доля — переобучить
      PIXGBM с PIXGBM_SAT_CORE_M2=60 и поставить [geophys.pixgbm] sat_core_max_m2 = 60 (одно значение
      в обучении и в сдаче).
H-G2. pick_rasters берёт серую копию, если она есть. На 211 серая карта сигнала не несёт (потолок 0,005),
      цветные — несут (0,37 и 0,50, notes/geo_probe_train.json). Выбирает ли лучшую копию правило без
      эталона «больше высокочастотной энергии в данных»? Печатается hp_energy каждой копии.

    python scripts/geo_sat_probe.py --root F:/2/train-expds-3 --gt F:/expds3/work/gt_train_v2.geojson \
        --out notes/geo_sat_probe_train.json

Имена участков в notes/ — только трейна. Эталон публичной выборки не нужен: без --gt печатаются доли
ядер и hp_energy, этого хватает для H-G2 на публичных картах.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expds import geo_pixgbm as G  # noqa: E402
from expds.catalog import read_utm_json, scan_root  # noqa: E402


def gt_geoms(gt: dict | None, region: str, markup: str, cls: str) -> list:
    from shapely.geometry import shape

    if gt is None:
        return []
    return [shape(f["geometry"]) for f in gt.get("features", [])
            if (f.get("properties") or {}).get("region_name") == region
            and f["properties"].get("markup_type") == markup and f["properties"].get("class_name") == cls]


def to_px(d: dict, geoms: list, pole: bool) -> np.ndarray:
    """Эталон EPSG:3857 → (строка, столбец) карты; для полигонов — полюс недоступности."""
    from shapely.ops import polylabel

    from expds.crs import WEB_MERCATOR, to_crs

    out = []
    inv = ~d["T"]
    for g in geoms:
        g = to_crs(g, WEB_MERCATOR, d["crs"]) if d["crs"] else g
        if pole and g.geom_type == "Polygon":
            try:
                c = polylabel(g, tolerance=0.02)
            except Exception:
                c = g.representative_point()
        else:
            c = g.centroid
        col, row = inv * (c.x, c.y)
        out.append((int(np.floor(row)), int(np.floor(col))))
    return np.asarray(out, np.int64).reshape(-1, 2)


def share_in(mask: np.ndarray, rc: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    ok = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
    rc = rc[ok]
    return int(mask[rc[:, 0], rc[:, 1]].sum()) if len(rc) else 0, int(len(rc))


def hp_energy(d: dict) -> float:
    """Медиана |z − среднее 2 м| в данных: сколько мелкой структуры несёт копия (без эталона)."""
    from scipy import ndimage as ndi

    z, v = d["z"], d["valid"]
    if v.sum() < 100:
        return 0.0
    k = max(3, int(round(2.0 / d["gsd"])) | 1)
    hp = np.abs(z - ndi.uniform_filter(np.where(v, z, 0), k))
    return round(float(np.median(hp[v])), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--core-m2", type=float, default=60.0, help="предел площади ядра, как plateau_max_m2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8")) if a.gt else None
    rows, tot = [], {"finds": [0, 0], "poly": [0, 0]}
    for reg in scan_root(a.root):
        crs = read_utm_json(reg.utm_json)
        for mt in ("maggr", "etgr", "terfl"):
            files = reg.rasters.get(mt, [])
            picked = set(G.pick_rasters(files))
            for f in files:
                rec = {"region": reg.name, "markup": mt, "file": f.name, "picked": f in picked}
                try:
                    d0 = G.load_map(f, G.GSD[mt], crs, sat_core_max_m2=0)
                    d1 = G.load_map(f, G.GSD[mt], crs, sat_core_max_m2=a.core_m2)
                    core = d1["valid"] & ~d0["valid"]
                    rec.update(core_share=round(float(core.mean()), 5), hp_energy=hp_energy(d0),
                               valid_share=round(float(d0["valid"].mean()), 4))
                    for key, cls, pole in (("finds", "finds_points", False), ("poly", "object_poly", True)):
                        rc = to_px(d0, gt_geoms(gt, reg.name, mt, cls), pole)
                        if len(rc):
                            hit, n = share_in(core, rc)
                            rec[f"gt_{key}"], rec[f"gt_{key}_in_core"] = n, hit
                            if f in picked:
                                tot[key][0] += hit
                                tot[key][1] += n
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(rec)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
    summary = {k: {"in_core": v[0], "gt": v[1], "share": round(v[0] / v[1], 4) if v[1] else None}
               for k, v in tot.items()}
    print(json.dumps({"summary_picked_maps": summary}, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=1),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
