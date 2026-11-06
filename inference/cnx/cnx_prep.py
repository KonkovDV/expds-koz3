"""Вход модели ConvNeXt-лидар (cnx3/cnx5) из одного LAS/LAZ: точная копия prep-цепочки обучения
(src/convnext_main.py:build_relief, стр. 284–292, + norm стр. 469). Свой код, только чтение вендора.

    import cnx_prep
    x, geo = cnx_prep.cnx_input("01_Осечки_1.3км_La_точки.laz", ch=5)
    # x: np.float32 [ch, H, W], значения ≈[-1, 1], вне съёмки -1.0079 (u8 0); geo: left/top/step/transform/crs/valid/...

CLI (отдельный процесс: у подготовки свои пакеты с теми же именами expds и expds_core):
    python cnx_prep.py LAS --ch 5 --out x.npz [--crs EPSG:32637] [--vendor DIR]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

VENDOR = Path(os.environ.get("CNX_VENDOR", Path(__file__).resolve().parent.parent / "kaggle/convnext_lidar/src/vendor"))
TAGS5 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")      # = контракт модели, c3 = первые три
# c12 (05.11, kaggle/2026-11-05/li_c12_in): те же функции вендора, что prep12.py; первые пять = TAGS5, dens_r/int_r — из LAS
TAGS12 = TAGS5 + ("opn_r", "tpi5_r", "tri3_r", "curv_r", "slrm8_r", "dens_r", "int_r")
CELL_M = 0.5                                                # шаг сети
DEM_RES = 0.1                                               # шаг ЦМР (среднее точек класса 2 + Delaunay)


def _import_vendor(vendor: Path):
    """Вендор первым в sys.path. Если expds / expds_core уже импортированы из другого места — ошибка
    (у те же имена модулей: expds/*.py совпадают побайтно, а expds_core.abi3.so — другая сборка)."""
    vendor = Path(vendor).resolve()
    for m in ("expds", "expds_core", "relief_ext"):
        mod = sys.modules.get(m)
        if mod is not None and not str(getattr(mod, "__file__", "")).startswith(str(vendor)):
            raise RuntimeError(f"{m} уже импортирован из {getattr(mod, '__file__', None)}; запускайте cnx_prep в отдельном процессе")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from expds import accel as XA, lidar as XL
    from expds.candidates import to_work_grid
    import relief_ext as RX
    return XA, XL, to_work_grid, RX


def cnx_u8(las_path, ch: int = 5, crs: str | None = None, vendor: Path = VENDOR):
    """LAS/LAZ (один файл = один участок) → (u8 [ch,H,W], geo). u8 0 = вне съёмки, 1…255 — значения."""
    assert ch in (3, 5, 12), ch
    XA, XL, to_work_grid, RX = _import_vendor(vendor)
    d = XL.build_dem(str(las_path), res=DEM_RES, fill="delaunay", crs=crs)   # сетка по заголовку LAS (все точки)
    base, tr = XL.relief_channels(d, CELL_M, TAGS5[:3])                     # Rust-путь (expds_core) или NumPy
    z, step = to_work_grid(d.dem, d.res, CELL_M)
    if z.shape != base.shape[1:]:
        raise RuntimeError(f"сетка z {z.shape} ≠ каналы {base.shape}")
    arr, stats = base, {}
    if ch == 5:
        z = np.where(base[0] > 0, z, np.nan).astype(np.float32)
        ext, stats = RX.extra_channels(z, step, d.left, d.top, accel=None, tags=TAGS5[3:])   # accel=None → rvt.vis (как в обучении)
        arr = np.concatenate([base, ext])
    elif ch == 12:
        z = np.where(base[0] > 0, z, np.nan).astype(np.float32)
        ext, stats = RX.extra_channels(z, step, d.left, d.top, las_path=str(las_path), accel=None, tags=TAGS12[3:])
        arr = np.concatenate([base, ext])
    arr = arr.astype(np.uint8)
    geo = dict(left=float(d.left), top=float(d.top), step=float(step), crs=d.crs, shape=list(arr.shape), tags=list(TAGS12 if ch == 12 else TAGS5[:ch]),
               transform=(float(step), 0.0, float(d.left), 0.0, -float(step), float(d.top)),   # affine.Affine(a,b,c,d,e,f), север вверху
               dem_res=float(d.res), dem_factor=int(d.info.get("factor", 1)), backend=XA.backend(), site_stats=stats,
               valid=(arr[0] > 0))
    return arr, geo


def u8_to_input(u8: np.ndarray) -> np.ndarray:
    """Нормировка обучения: x = (u8 − 128) / 127 (float32)."""
    return (u8.astype(np.float32) - 128.0) / 127.0


def cnx_input(las_path, ch: int = 5, crs: str | None = None, vendor: Path = VENDOR):
    """LAS/LAZ → (x float32 [ch,H,W], geo). Пиксель (r, c) покрывает [left+c·step, left+(c+1)·step) × (top−(r+1)·step, top−r·step]."""
    u8, geo = cnx_u8(las_path, ch, crs, vendor)
    return u8_to_input(u8), geo


if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser()
    ap.add_argument("las"); ap.add_argument("--ch", type=int, default=5); ap.add_argument("--crs", default=None)
    ap.add_argument("--vendor", default=str(VENDOR)); ap.add_argument("--out", required=True)
    a = ap.parse_args(); t = time.time()
    u8, geo = cnx_u8(a.las, a.ch, a.crs, Path(a.vendor))
    valid = geo.pop("valid")
    np.savez_compressed(a.out, u8=u8, valid=valid, geo=json.dumps(geo, ensure_ascii=False, default=str))
    print(json.dumps(dict(geo, t_s=round(time.time() - t, 1)), ensure_ascii=False, default=str))
