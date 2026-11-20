#!/usr/bin/env python3
"""Двенадцать каналов рельефа (вход c12-in) из LAS: lrm_s, lrm_l, slope и девять каналов relief_ext.

Тот же расчёт, которым 05.11 построены входы обучения c12-in. Код каналов — вендор `inference/cnx/vendor` (тот же, что в ядре обучения: `vendor/vendor_manifest.json`)
и RVT 2.2.1 из `inference/cnx/_deps`. Первые пять каналов побайтно равны входу c5 (те же функции вендора,
accel=None → rvt); инференс строит тот же вход функцией `inference/cnx/cnx_prep.py` (ch = 12).

    python training/lidar_convnext/prep12.py LAS CRS OUT_STEM          # один LAS → OUT_STEM.npy и OUT_STEM.json
    python training/lidar_convnext/prep12.py --root DATA --out work/relief12   # все участки с LAS (*_La_*.las)

CRS — поле `crs` из UTM.json участка (как в ядре обучения); пустая строка — CRS из заголовка LAS.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "inference" / "cnx" / "vendor"
TAGS12 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r", "opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")
CELL_M, DEM_RES = 0.5, 0.1


@functools.cache
def _vendor():
    """Модули вендора первыми в sys.path (у пайплайна есть пакет с тем же именем expds)."""
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(REPO / "inference" / "cnx" / "_deps"))
    from expds import accel as XA, lidar as XL
    from expds.candidates import to_work_grid
    import relief_ext as RX
    return XA, XL, to_work_grid, RX


def prep(las: Path, crs: str | None, out: Path) -> tuple[tuple[int, ...], float]:
    """Каналы одного LAS → out.npy (u8 [12, H, W]) и out.json (геопривязка, теги, статистики нормировки)."""
    XA, XL, to_work_grid, RX = _vendor()
    t = time.time()
    d = XL.build_dem(str(las), res=DEM_RES, fill="delaunay", crs=crs)
    base, _tr = XL.relief_channels(d, CELL_M, TAGS12[:3])
    z, step = to_work_grid(d.dem, d.res, CELL_M)
    z = np.where(base[0] > 0, z, np.nan).astype(np.float32)
    ext, stats = RX.extra_channels(z, step, d.left, d.top, las_path=str(las), accel=None, tags=TAGS12[3:])
    arr = np.concatenate([base, ext]).astype(np.uint8)
    geo = dict(left=float(d.left), top=float(d.top), step=float(step), crs=d.crs, shape=list(arr.shape), tags=list(TAGS12),
               dem_res=float(d.res), dem_factor=int(d.info.get("factor", 1)), backend=XA.backend(), site_stats=stats)
    # имя участка может содержать точку («006_Осечки_1.1км»): суффикс дописывается, а не заменяется
    np.save(out.parent / (out.name + ".npy"), arr)
    (out.parent / (out.name + ".json")).write_text(json.dumps(geo, ensure_ascii=False, default=str))
    return arr.shape, round(time.time() - t, 1)


def site_inputs(root: Path) -> list[tuple[str, Path, str | None]]:
    """(участок, LAS, CRS из UTM.json) для каждого участка выборки с лидаром — как region_inv ядра обучения."""
    out = []
    for site in sorted(p for p in root.iterdir() if p.is_dir()):
        las = sorted(p for p in site.rglob("*") if p.is_file() and p.suffix.lower() == ".las" and "_La_" in p.name)
        if not las:
            continue
        utm = site / "UTM.json"
        crs = (json.loads(utm.read_text(encoding="utf-8")).get("crs") or None) if utm.is_file() else None
        out.append((site.name, las[-1], crs))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("las", nargs="?", type=Path)
    ap.add_argument("crs", nargs="?", default="")
    ap.add_argument("out_stem", nargs="?", type=Path)
    ap.add_argument("--root", type=Path, help="папка обучающей выборки (участки — подпапки)")
    ap.add_argument("--out", type=Path, help="папка выхода для --root")
    a = ap.parse_args(argv)
    if a.root:
        if not a.out:
            ap.error("--root требует --out")
        a.out.mkdir(parents=True, exist_ok=True)
        for site, las, crs in site_inputs(a.root):
            if (a.out / f"{site}.npy").exists():
                print(site, "skip", flush=True)
                continue
            print(json.dumps({"site": site, "las": las.name, "shape_s": prep(las, crs, a.out / site)}, ensure_ascii=False),
                  flush=True)
        return 0
    if not (a.las and a.out_stem):
        ap.error("нужны LAS CRS OUT_STEM или --root DATA --out DIR")
    print(json.dumps({"las": a.las.name, "shape_s": prep(a.las, a.crs or None, a.out_stem)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
