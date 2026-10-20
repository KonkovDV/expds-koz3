"""LAS → DEM → стандартный набор отмывок (c, ch, g, i), как у организаторов.

В валидационной выборке готовых карт Li нет: решение строит их само.
Разметка Li делалась по отмывкам, построенным алгоритмом бейзлайна, поэтому
здесь он повторён шаг в шаг (параметры — в DEFAULTS). Проверка совпадения
с картами трейна: `python -m expds.lidar check --las X.las --li-dir папка_Li`.

Командная строка:
    python -m expds.lidar build --las X.las --out D:/work/li [--res 0.1]
    python -m expds.lidar check --las X.las --li-dir D:/expds3/001_.../02_..._Li_карты
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import accel

log = logging.getLogger(__name__)

DEFAULTS = {
    "res": 0.1,              # шаг DEM, м
    "max_pixels": 60_000_000,  # больше ячеек → шаг увеличивается (как у бейзлайна)
    "ground_class": 2,
    "max_vertices": 4_000_000,  # до этого числа занятых ячеек — Делоне (scipy)
    "max_vertices_rust": 20_000_000,  # то же для Rust (≈ 72 байта на вершину)
    "cycles_c": 3.0,
}
#: DEM для моделей и кандидатов: TIN по ячейкам 0,1 м, затем усреднение до рабочего шага.
#: Вариант «сразу 0,5 м + пирамида» быстрее, но хуже: детектор курганов вне обучения 0,373 против 0,411.
MODEL_DEM = {"res": 0.1, "fill": "delaunay"}
#: DEM моделей без dem_res в чекпойнте (обучены до появления параметра) — тот же TIN 0,1 м.
LEGACY_DEM = {"res": 0.1, "fill": "delaunay"}
TAGS = ("c", "ch", "g", "i")
#: Каналы модели li, построенные из LAS (relief_channels). Не зависят от освещения.
RELIEF_TAGS = ("lrm_s", "lrm_l", "slope")
#: Радиусы окна локального рельефа (м) и шкалы перевода в uint8.
RELIEF = {"lrm_s_m": 8.0, "lrm_s_clip_m": 0.75, "lrm_l_m": 30.0, "lrm_l_clip_m": 2.0, "slope_max_deg": 45.0}


@dataclass
class Dem:
    """Модель рельефа с привязкой. dem — float32, NaN вне контура съёмки."""
    dem: np.ndarray
    left: float
    top: float
    res: float
    crs: str | None = None
    info: dict = field(default_factory=dict)

    @property
    def transform(self):
        """Аффинное преобразование rasterio (север вверху)."""
        from rasterio.transform import from_origin

        return from_origin(self.left, self.top, self.res, self.res)


def _fill_by_delaunay(dem: np.ndarray, valid: np.ndarray, to_fill: np.ndarray,
                      max_vertices: int) -> np.ndarray | None:
    """Линейная интерполяция по Делоне, как в бейзлайне; None — если не применима.

    С Rust — expds_core.fill_tin (до DEFAULTS["max_vertices_rust"] вершин), иначе scipy (Qhull)
    до max_vertices вершин.
    """
    n = int(valid.sum())
    if n <= DEFAULTS["max_vertices_rust"]:
        tin = accel.fill_tin(dem)
        if tin is not None:
            return np.where(to_fill, tin, dem).astype(np.float32)
    if n > max_vertices:
        return None
    try:
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay

        rows, cols = np.nonzero(valid)
        tri = Delaunay(np.column_stack([cols, rows]).astype(np.float64))
        interp = LinearNDInterpolator(tri, dem[rows, cols])
        filled = dem.copy()
        fr, fc = np.nonzero(to_fill)
        for s in range(0, fr.size, 2_000_000):
            sl = slice(s, s + 2_000_000)
            vals = interp(np.column_stack([fc[sl], fr[sl]]).astype(np.float64))
            ok = np.isfinite(vals)
            filled[fr[sl][ok], fc[sl][ok]] = vals[ok]
        return filled
    except Exception as exc:
        log.debug("Делоне не сработал: %s", exc)
        return None


def fill_interior_holes(dem: np.ndarray, max_vertices: int = DEFAULTS["max_vertices"],
                        method: str = "delaunay") -> np.ndarray:
    """Заполняет пустоты внутри выпуклой оболочки съёмки; снаружи остаётся NaN.

    method: delaunay — линейная интерполяция по Делоне, как у бейзлайна (пирамида, если ячеек
    больше max_vertices); pyramid — многомасштабная пирамида (pull-push).
    """
    valid = np.isfinite(dem)
    if valid.all():
        return dem
    to_fill = accel.convex_hull_mask(valid) & ~valid
    if not to_fill.any():
        return dem
    filled = _fill_by_delaunay(dem, valid, to_fill, max_vertices) if method == "delaunay" else None
    if filled is None:
        filled = np.where(to_fill, accel.fill_pyramid(dem), dem)
    unresolved = to_fill & ~np.isfinite(filled)
    if unresolved.any():
        filled = np.where(unresolved, accel.fill_pyramid(filled), filled)
    return filled.astype(np.float32)


def build_dem(las_path: str | Path, res: float = DEFAULTS["res"],
              max_pixels: int = DEFAULTS["max_pixels"], ground_class: int = DEFAULTS["ground_class"],
              crs: str | None = None, fill: str = "delaunay") -> Dem:
    """Строит DEM из LAS/LAZ: сетка средних высот точек земли + заполнение пустот (fill)."""
    t0 = time.perf_counter()
    dem, info = accel.las_grid(las_path, res, ground_class, max_pixels)
    t1 = time.perf_counter()
    holes = int(np.count_nonzero(~np.isfinite(dem)))
    dem = fill_interior_holes(dem, method=fill)
    t2 = time.perf_counter()
    info = dict(info, holes=holes, t_grid_s=round(t1 - t0, 2), t_fill_s=round(t2 - t1, 2),
                backend=accel.backend())
    if crs is None:
        from .crs import las_crs

        crs = las_crs(las_path)
    log.info("DEM %s: %dx%d @ %.3g м, точек %s, пустот %d, %s", Path(las_path).name,
             dem.shape[1], dem.shape[0], info["res"], info["points_used"], holes, info["backend"])
    return Dem(dem=dem, left=info["left"], top=info["top"], res=info["res"], crs=crs, info=info)


class DemPrefetch:
    """DEM следующих LAS строятся в фоновых потоках, пока текущий участок считается.

    Сетка, TIN и заливка — Rust без GIL, так что потоки действительно параллельны
    основному: тот в это время ждёт видеокарту и собирает полигоны. Чтение
    холодного файла с диска тоже уходит в фон. Триангуляция Делоне идёт в одном
    потоке, поэтому depth облаков триангулируются одновременно на разных ядрах.
    В памяти — не больше depth + 1 DEM.
    """

    def __init__(self, order: list[tuple[Path, str | None]], depth: int = 2):
        from concurrent.futures import ThreadPoolExecutor

        self.depth = max(1, int(depth))
        self._ahead = {str(p): order[i + 1:i + 1 + self.depth] for i, (p, _) in enumerate(order)}
        self._pool = ThreadPoolExecutor(max_workers=self.depth, thread_name_prefix="dem")
        self._jobs: dict[tuple, object] = {}

    def _submit(self, las: Path, res: float, fill: str, crs: str | None) -> tuple:
        key = (str(las), float(res), fill, crs)
        if key not in self._jobs:
            self._jobs[key] = self._pool.submit(build_dem, las, res=res, fill=fill, crs=crs)
        return key

    def get(self, las: Path, res: float, fill: str, crs: str | None) -> Dem:
        """DEM этого LAS; следующие depth по порядку ставятся в очередь сразу за ним."""
        key = self._submit(las, res, fill, crs)
        keep = {key} | {self._submit(p, res, fill, c) for p, c in self._ahead.get(str(las), [])}
        # LAS, пропущенный по времени, не должен держать свой DEM в памяти.
        for stale in [k for k in self._jobs if k not in keep]:
            self._jobs.pop(stale).cancel()
        return self._jobs.pop(key).result()

    def close(self) -> None:
        for fut in self._jobs.values():
            fut.cancel()
        self._jobs.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


def dem_params(meta: dict | None = None, cfg: dict | None = None) -> dict:
    """Параметры DEM: из чекпойнта модели li (как при обучении), иначе из [lidar] конфига, иначе MODEL_DEM."""
    if meta is not None:
        if meta.get("dem_res") is None:
            return dict(LEGACY_DEM)
        return {"res": float(meta["dem_res"]), "fill": meta.get("dem_fill") or MODEL_DEM["fill"]}
    lid = (cfg or {}).get("lidar", {})
    return {"res": float(lid.get("dem_res", MODEL_DEM["res"])), "fill": lid.get("dem_fill", MODEL_DEM["fill"])}


def hillshade_set(d: Dem, tags=TAGS, cycles_c: float = DEFAULTS["cycles_c"],
                  cycles_ch: float | None = None) -> dict[str, np.ndarray]:
    """Отмывки по DEM: g, i — H×W uint8 (nodata 255); c, ch — 3×H×W uint8 (nodata 0)."""
    valid = np.isfinite(d.dem)
    if not valid.any():
        raise ValueError("DEM пуст")
    z_min = float(np.nanmin(d.dem))
    z_max = float(np.nanmax(d.dem))
    z_range = max(z_max - z_min, 1e-9)
    filled = np.where(valid, d.dem, np.float32(z_min)).astype(np.float32)
    out: dict[str, np.ndarray] = {}
    if "g" in tags or "i" in tags:
        shade = accel.hillshade(filled, d.res, d.res)
        if "g" in tags:
            g = shade.copy()
            g[~valid] = 255
            out["g"] = g
        if "i" in tags:
            i = (255 - shade.astype(np.int16)).astype(np.uint8)
            i[~valid] = 255
            out["i"] = i
    cycles = {"c": float(cycles_c),
              "ch": float(cycles_ch) if cycles_ch else float(max(1, round(2.0 * z_range)))}
    for tag in ("c", "ch"):
        if tag in tags:
            out[tag] = accel.color_relief(filled, valid, z_min, z_range, cycles[tag])
    return out


def slope_deg(z: np.ndarray, res: float) -> np.ndarray:
    """Уклон в градусах по ядру Хорна 3×3 (как в ArcGIS); на краях — повтор крайних пикселей."""
    return accel.slope_deg(z, res)


def relief_channels(d: Dem, res: float, tags=RELIEF_TAGS):
    """Каналы модели li из DEM на рабочем шаге: C×H×W uint8 и привязка (север вверху).

    Одна и та же функция строит кэш тайлов и вход модели на инференсе: в
    валидации готовых карт Li нет, а карты Li трейна построены другим
    инструментом и не повторяются по LAS (expds.lidar check). Значения 1…255,
    0 — вне контура съёмки.
    lrm_s, lrm_l — локальный рельеф (высота − среднее окна радиусом 8 и 30 м,
    под курганы лесной и степной зон); slope — уклон; g — серая отмывка.
    Только g зависит от направления света: без него допустимы повороты.
    """
    from rasterio.transform import from_origin

    from .candidates import to_work_grid

    rs = accel._rs
    if rs is not None and hasattr(rs, "relief_channels"):
        try:
            img, step = rs.relief_channels(
                np.ascontiguousarray(d.dem, np.float32), float(d.res), float(res), list(tags),
                float(RELIEF["lrm_s_m"]), float(RELIEF["lrm_s_clip_m"]),
                float(RELIEF["lrm_l_m"]), float(RELIEF["lrm_l_clip_m"]),
                float(RELIEF["slope_max_deg"]))
            step = float(step)
            return img, from_origin(d.left, d.top, step, step)
        except Exception as exc:
            log.debug("каналы li на Rust не собраны (%s), считаю в NumPy", exc)
    z, step = to_work_grid(d.dem, d.res, res)
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.float32(np.nanmean(z)) if valid.any() else np.float32(0)).astype(np.float32)
    out = []
    for t in tags:
        if t in ("lrm_s", "lrm_l"):
            radius = max(1, round(RELIEF[f"{t}_m"] / step))
            lr = accel.local_relief(np.where(valid, z, np.nan).astype(np.float32), radius)
            unit = np.clip(np.nan_to_num(lr, nan=0.0) / RELIEF[f"{t}_clip_m"], -1, 1) * 0.5 + 0.5
        elif t == "slope":
            unit = np.clip(slope_deg(filled, step) / RELIEF["slope_max_deg"], 0, 1)
        elif t == "g":
            unit = accel.hillshade(filled, step, step).astype(np.float32) / 255.0
        else:
            raise ValueError(f"неизвестный канал li: {t}")
        out.append(np.where(valid, np.rint(1 + unit * 254), 0).astype(np.uint8))
    return np.stack(out), from_origin(d.left, d.top, step, step)


def write_set(images: dict[str, np.ndarray], d: Dem, out_dir: str | Path, name: str) -> list[Path]:
    """Сохраняет отмывки в GeoTIFF с привязкой DEM (для QGIS и проверки)."""
    import rasterio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tag, arr in images.items():
        a = arr if arr.ndim == 3 else arr[None]
        p = out_dir / f"{name}_{tag}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                           dtype="uint8", transform=d.transform, crs=d.crs, compress="deflate") as ds:
            ds.write(a)
        paths.append(p)
    return paths


def compare_with_li(images: dict[str, np.ndarray], d: Dem, li_dir: str | Path) -> dict:
    """Сравнивает построенные отмывки с картами Li трейна (по пересечению экстентов).

    Для каждого тега: размер, шаг, доля совпавших пикселей и доля |Δ| ≤ 1.
    Совпадение > 99 % означает, что модели, обученные на Li трейна,
    увидят на валидации те же картинки.
    """
    import rasterio
    from rasterio.windows import from_bounds

    report = {}
    for tag, ours in images.items():
        cands = sorted(Path(li_dir).glob(f"*_{tag}.tif"))
        if not cands:
            report[tag] = {"status": "нет файла Li"}
            continue
        with rasterio.open(cands[0]) as ds:
            res_li = ds.res[0]
            if abs(res_li - d.res) > 1e-6:
                report[tag] = {"status": "разный шаг", "li_res": res_li, "our_res": d.res}
                continue
            ours3 = ours if ours.ndim == 3 else ours[None]
            h, w = ours3.shape[1:]
            bounds = (d.left, d.top - h * d.res, d.left + w * d.res, d.top)
            win = from_bounds(*bounds, transform=ds.transform).round_offsets().round_lengths()
            theirs = ds.read(window=win, boundless=True, fill_value=0)
        hh, ww = min(theirs.shape[1], h), min(theirs.shape[2], w)
        a = ours3[:, :hh, :ww].astype(np.int16)
        b = theirs[: a.shape[0], :hh, :ww].astype(np.int16)
        diff = np.abs(a - b)
        dx, dy, peak = phase_shift(a[0].astype(np.float32), b[0].astype(np.float32))
        report[tag] = {"status": "ok", "file": cands[0].name, "pixels": int(diff[0].size),
                       "equal_pct": round(100 * float((diff == 0).mean()), 3),
                       "within1_pct": round(100 * float((diff <= 1).mean()), 3),
                       "mae": round(float(diff.mean()), 4),
                       # сдвиг Li относительно наших отмывок: x — на восток, y — на юг (пиксели и метры)
                       "shift_px": [dx, dy], "shift_m": [round(dx * d.res, 3), round(dy * d.res, 3)],
                       "shift_peak": peak}
    return report


def phase_shift(a: np.ndarray, b: np.ndarray, max_side: int = 2048) -> tuple[float, float, float]:
    """Сдвиг b относительно a методом фазовой корреляции (целые пиксели исходного шага).

    Нужен, когда совпадение низкое: отличает «сдвинута геопривязка» (чёткий пик,
    сдвиг ≠ 0) от «другая модель рельефа» (пик размытый, сдвиг ≈ 0).
    """
    step = max(1, int(np.ceil(max(a.shape) / max_side)))
    a, b = a[::step, ::step], b[::step, ::step]
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    fa, fb = np.fft.rfft2((a - a.mean()) * win), np.fft.rfft2((b - b.mean()) * win)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.irfft2(cross, s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = iy - a.shape[0] if iy > a.shape[0] // 2 else iy
    dx = ix - a.shape[1] if ix > a.shape[1] // 2 else ix
    return float(dx * step), float(dy * step), round(float(corr.max()), 3)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="LAS → DEM → отмывки c/ch/g/i (как у организаторов)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "check"):
        p = sub.add_parser(name)
        p.add_argument("--las", required=True)
        p.add_argument("--res", type=float, default=DEFAULTS["res"])
        p.add_argument("--max-pixels", type=int, default=DEFAULTS["max_pixels"])
        p.add_argument("--crs", default=None, help="EPSG, если в LAS нет CRS (например, из UTM.json)")
        if name == "build":
            p.add_argument("--out", required=True)
        else:
            p.add_argument("--li-dir", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.perf_counter()
    d = build_dem(args.las, args.res, args.max_pixels, crs=args.crs)
    images = hillshade_set(d)
    t1 = time.perf_counter()
    name = Path(args.las).stem.replace("_La_точки", "").replace("_La", "").replace("_ground", "")
    if args.cmd == "build":
        paths = write_set(images, d, args.out, name)
        print(json.dumps({"files": [str(p) for p in paths], "seconds": round(t1 - t0, 1),
                          "info": {k: v for k, v in d.info.items() if k != "classes"}},
                         ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"seconds": round(t1 - t0, 1), "report": compare_with_li(images, d, args.li_dir)},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
