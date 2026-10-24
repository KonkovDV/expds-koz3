#!/usr/bin/env python3
"""Аудит растров трейна: CRS, шаг на земле, привязка по GCP, поворот, тип данных.

Зачем. Модели снимков учатся и работают на «шаге в единицах CRS». В UTM это
метры, в EPSG:3857 — метры, растянутые в 1/cos φ раз (на 55° с. ш. в 1,74 раза), в
градусах — вообще не метры. Если в трейне и в выборке доли таких снимков разные,
масштаб насыпи на входе сети разный. Таблица отвечает: нужен ли пересчёт шага в
метры земли при обучении и инференсе (план v5, E-G2) и сколько снимков сейчас
читаются через перепроецирование (raster.open_metric).

    python scripts/audit_rasters.py --train F:/expds3/train-expds-3 --out notes/raster_audit_2026-10-24.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def describe(path: Path) -> dict:
    import rasterio

    from expds.crs import is_geographic
    from expds.raster import needs_warp

    with rasterio.open(path) as ds:
        crs = ds.crs.to_string() if ds.crs else None
        epsg = ds.crs.to_epsg() if ds.crs else None
        kind = ("gcps" if crs is None and ds.gcps[0] else "none" if crs is None else
                "geographic" if is_geographic(crs) else "3857" if epsg == 3857 else
                "utm" if epsg and str(epsg).startswith(("326", "327")) else "other")
        res = float(abs(ds.res[0]))
        ground = res
        if kind == "3857":
            b = ds.bounds
            lat = math.degrees(math.atan(math.sinh(((b.top + b.bottom) / 2) / 6378137.0)))
            ground = res * math.cos(math.radians(lat))
        elif kind == "geographic":
            b = ds.bounds
            _, lat = (b.left + b.right) / 2, (b.top + b.bottom) / 2
            ground = res * 111_320 * math.cos(math.radians(lat))
        rotated = abs(ds.transform.b) > 1e-12 or abs(ds.transform.d) > 1e-12
        return {"file": path.name, "kind": kind, "epsg": epsg, "res_crs": res, "res_ground_m": round(ground, 4),
                "rotated": rotated, "warp": needs_warp(ds), "bands": ds.count, "dtype": ds.dtypes[0],
                "width": ds.width, "height": ds.height, "nodata": ds.nodata,
                "mb": round(path.stat().st_size / 2 ** 20, 1)}


def main() -> None:
    from expds.catalog import scan_root

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for reg in scan_root(a.train):
        for mt, files in reg.rasters.items():
            for f in files:
                try:
                    rows.append({"region": reg.name, "markup": mt, **describe(f)})
                except Exception as exc:
                    rows.append({"region": reg.name, "markup": mt, "file": f.name, "error": repr(exc)})
    by = defaultdict(list)
    for r in rows:
        by[r["markup"]].append(r)
    summary = {}
    for mt, rs in sorted(by.items()):
        ground = sorted(r["res_ground_m"] for r in rs if "res_ground_m" in r)
        summary[mt] = {"files": len(rs), "kinds": dict(Counter(r.get("kind", "error") for r in rs)),
                       "warp": dict(Counter(str(r.get("warp")) for r in rs)),
                       "rotated": sum(1 for r in rs if r.get("rotated")),
                       "ground_res_m_p10_p50_p90": [ground[int(q * (len(ground) - 1))] for q in (0.1, 0.5, 0.9)]
                       if ground else None}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "files": rows}, ensure_ascii=False, indent=2),
                               encoding="utf-8")


if __name__ == "__main__":
    main()
