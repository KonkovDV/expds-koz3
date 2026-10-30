"""LAS → DEM → стандартный набор отмывок (c, ch, g, i), как у организаторов.

В валидационной выборке готовых карт Li нет: решение строит их само.
Разметка Li делалась по отмывкам, построенным алгоритмом бейзлайна, поэтому
здесь он повторён шаг в шаг (параметры — в DEFAULTS). Проверка совпадения
с картами трейна: `python -m expds.lidar check --las X.las --li-dir папка_Li`.

Командная строка:
    python -m expds.lidar build --las X.las --out <папка> [--res 0.1]
    python -m expds.lidar check --las X.las --li-dir <данные>/<участок>/<Li_карты>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
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
#: Набор для крупных контуров под лесом: локальный рельеф, уклон, отмывка, шероховатость,
#: кривизна. Отмывка — не производная поверхности и зависит от солнца. Высота заменена
#: локальным рельефом: сырая высота тащит макрорельеф. В сданные веса не входит.
ENCLOSURE_TAGS = ("lrm_l", "slope", "g", "rough", "curv")
#: Радиусы окна локального рельефа (м) и шкалы перевода в uint8.
#: opns, svf — положительная открытость и фактор вида на небо: луч opns_r_m, opns_dirs направлений,
#: линейная шкала в пределах *_range (1–99 перцентили на шаге 0,5 м по четырём участкам трейна).
RELIEF = {"lrm_s_m": 8.0, "lrm_s_clip_m": 0.75, "lrm_l_m": 30.0, "lrm_l_clip_m": 2.0,
          "lrm_xl_m": 100.0, "lrm_xl_clip_m": 4.0, "slope_max_deg": 45.0,
          "opns_r_m": 10.0, "opns_dirs": 16, "opns_range": (75.0, 92.0), "svf_range": (0.75, 1.0),
          "dev_s_m": 8.0, "dev_l_m": 30.0, "dev_xl_m": 100.0,
          "dev_clip": 3.0, "dev_std_floor_m": 0.05,
          "rough_hp_m": 1.0, "rough_win_m": 1.5, "rough_clip_m": 0.25,
          "curv_sigma_m": 1.5, "curv_clip": 0.1}
#: Каналы, которые Rust собирает одним вызовом (expds_core.relief_channels).
FUSED_TAGS = {"lrm_s", "lrm_l", "slope", "g"}
#: Канал int — яркость отражения лазера (интенсивность) точек земли: шкала по перцентилям облака,
#: пустые ячейки — среднее окна INTENSITY_FILL_M. Сданные веса его не используют.
INTENSITY_PCT = (2.0, 98.0)
INTENSITY_FILL_M = 2.0


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
            return np.where(to_fill, tin, dem).astype(np.float32, copy=False)
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
    return filled.astype(np.float32, copy=False)


LOWVEG_FILL = "tin_lowveg"   # P-W2-10: lidar.dem_fill="tin_lowveg"; по умолчанию "delaunay" (как было)
LOWVEG = dict(classes=(1, 3), parent_m=0.5, open_r_m=1.5, dz_lo=-0.5, dz_hi=3.0)


def fill_lowveg(dem: np.ndarray, las_path: str | Path, info: dict, max_pixels: int = DEFAULTS["max_pixels"],
                ground_class: int = 2, base: str = "delaunay", p: dict = LOWVEG) -> tuple[np.ndarray, dict]:
    """Дыры класса 2 → вершины из нижних отражений классов 1/3 (P-W2-10, PREREG_W2_cnx.md).

    N = поклеточный min средних z классов 1 и 3 на сетке DEM; Lmin — min N в родителе 0,5 м; Lo — серое открытие Lmin
    диском r=1,5 м. Родитель без точек класса 2 с −0,5 ≤ Lo − T0 ≤ +3 м (T0 — обычная заливка base) даёт вершину Lo
    в центральной ячейке DEM внутри оболочки; затем та же fill_interior_holes(base). Без классов 1/3 — как base.
    """
    from scipy import ndimage as ndi

    res = float(info["res"])
    t0 = fill_interior_holes(dem, method=base)
    low = None
    for c in p["classes"]:
        g, gi = accel.las_grid(las_path, res, int(c), max_pixels)
        if g.shape != dem.shape or abs(float(gi["res"]) - res) > 1e-9:
            raise RuntimeError(f"lowveg: сетка класса {c} {g.shape}@{gi['res']} ≠ DEM {dem.shape}@{res}")
        low = g if low is None else np.fmin(low, g)
    stats = dict(n_accept=0, n_parent_holes=0, params=dict(p, classes=list(p["classes"])))
    if low is None or not np.isfinite(low).any():
        return t0, stats
    h, w = dem.shape
    pm = float(p["parent_m"])
    ri = np.minimum(((np.arange(h) + 0.5) * res // pm).astype(np.int64), int(h * res // pm + 1))
    ci = np.minimum(((np.arange(w) + 0.5) * res // pm).astype(np.int64), int(w * res // pm + 1))
    ph, pw = int(ri.max()) + 1, int(ci.max()) + 1
    lmin = np.full((ph, pw), np.inf, np.float32)
    hasg = np.zeros((ph, pw), bool)
    for r0 in range(0, h, 2048):  # построчными блоками: без копии сетки на int64
        sl = slice(r0, min(h, r0 + 2048))
        R = np.broadcast_to(ri[sl, None], (sl.stop - sl.start, w)).ravel()
        C = np.broadcast_to(ci[None, :], (sl.stop - sl.start, w)).ravel()
        lv = low[sl].ravel(); fin = np.isfinite(lv)
        np.minimum.at(lmin, (R[fin], C[fin]), lv[fin])
        gv = np.isfinite(dem[sl]).ravel()
        hasg[R[gv], C[gv]] = True
    rad = int(round(p["open_r_m"] / pm))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = (yy * yy + xx * xx) <= rad * rad
    lo = ndi.grey_opening(lmin, footprint=disk, mode="nearest")
    lo[~np.isfinite(lmin)] = np.inf
    # центральная ячейка DEM каждого родителя
    cr = np.minimum(((np.arange(ph) + 0.5) * pm / res).astype(np.int64), h - 1)
    cc = np.minimum(((np.arange(pw) + 0.5) * pm / res).astype(np.int64), w - 1)
    CR, CC = np.meshgrid(cr, cc, indexing="ij")
    t0c = t0[CR, CC]
    dz = lo - t0c
    ok = (~hasg) & np.isfinite(lo) & np.isfinite(t0c) & (dz >= p["dz_lo"]) & (dz <= p["dz_hi"]) & ~np.isfinite(dem[CR, CC])
    stats.update(n_parent_holes=int((~hasg & np.isfinite(t0c)).sum()), n_accept=int(ok.sum()),
                 n_lowveg_parents=int((~hasg & np.isfinite(lo) & np.isfinite(t0c)).sum()),
                 dz_accept_med=float(np.median(dz[ok])) if ok.any() else None)
    if not ok.any():
        return t0, stats
    d2 = dem.copy()
    d2[CR[ok], CC[ok]] = lo[ok]
    return fill_interior_holes(d2, method=base), stats


def thin_keep(name: str, seed: int = 0) -> float:
    """С вероятностью 0,5 оставить 25–100 % точек земли, иначе все.

    Доля зависит только от имени файла и зерна, чтобы кэш повторялся.
    На инференсе не вызывается: там облако читается целиком.
    """
    rng = random.Random(f"{seed}:{name}")
    if rng.random() >= 0.5:
        return 1.0
    return rng.uniform(0.25, 1.0)


def build_dem(las_path: str | Path, res: float = DEFAULTS["res"],
              max_pixels: int = DEFAULTS["max_pixels"], ground_class: int = DEFAULTS["ground_class"],
              crs: str | None = None, fill: str = "delaunay", keep_prob: float = 1.0) -> Dem:
    """Строит DEM из LAS/LAZ: сетка средних высот точек земли + заполнение пустот (fill)."""
    t0 = time.perf_counter()
    dem, info = accel.las_grid(las_path, res, ground_class, max_pixels, keep_prob=keep_prob)
    t1 = time.perf_counter()
    holes = int(np.count_nonzero(~np.isfinite(dem)))
    if fill == LOWVEG_FILL:
        dem, lv = fill_lowveg(dem, las_path, info, max_pixels, ground_class)
        info = dict(info, lowveg=lv)
    else:
        dem = fill_interior_holes(dem, method=fill)
    t2 = time.perf_counter()
    info = dict(info, holes=holes, t_grid_s=round(t1 - t0, 2), t_fill_s=round(t2 - t1, 2),
                backend=accel.backend(), las_path=str(las_path), ground_class=ground_class)
    if crs is None:
        from .crs import las_crs

        crs = las_crs(las_path)
    log.info("DEM %s: %dx%d @ %.3g м, точек %s, пустот %d, %s", Path(las_path).name,
             dem.shape[1], dem.shape[0], info["res"], info["points_used"], holes, info["backend"])
    return Dem(dem=dem, left=info["left"], top=info["top"], res=info["res"], crs=crs, info=info)


def would_coarsen(las_path: str | Path, res: float, max_pixels: int) -> bool:
    """True, если сетка всего облака на шаге res больше max_pixels и шаг пришлось бы увеличить."""
    import laspy

    from .accel import grid_spec

    with laspy.open(str(las_path)) as f:
        h = f.header
        spec = grid_spec(float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1]),
                         res, max_pixels)
    return int(spec["factor"]) > 1


def has_class(las_path: str | Path, ground_class: int = 2) -> bool:
    """Есть ли в облаке хотя бы одна точка этого класса. Читает, пока не найдёт."""
    import laspy

    with laspy.open(str(las_path)) as f:
        for pts in f.chunk_iterator(1_000_000):
            if np.any(np.asarray(pts.classification) == ground_class):
                return True
    return False


def build_dem_mosaic(las_path: str | Path, res: float = 0.1, work_res: float = 0.5,
                     tile_m: float = 400.0, halo_m: float = 40.0,
                     ground_class: int = 2, fill: str = "delaunay",
                     crs: str | None = None) -> Dem:
    """DEM шага work_res из кусков res, без огрубления всего облака.

    Крупный LAS не влезает в сетку 0,1 м, и общий шаг становится 0,2–0,3 м.
    Модель училась на среднем блоков 0,1 → 0,5 м. Здесь каждый кусок строится
    на 0,1 м с полем вокруг, пустоты заполняются, в мозаику попадает только
    середина, усреднённая до 0,5 м. Файл читается один раз. В конфиг не включено:
    на 24 грубых облаках это ещё не проверено официальной метрикой.
    """
    import laspy

    from .accel import grid_spec
    from .candidates import downsample_mean

    with laspy.open(str(las_path)) as f:
        hdr = f.header
        min_x, min_y = float(hdr.mins[0]), float(hdr.mins[1])
        max_x, max_y = float(hdr.maxs[0]), float(hdr.maxs[1])
    spec = grid_spec(min_x, min_y, max_x, max_y, res, max_pixels=0)
    k = int(round(work_res / res))
    if k < 1 or abs(res * k - work_res) > 1e-6:
        raise ValueError(f"шаг {work_res} не кратен {res}")
    tile_px = max(k, int(round(tile_m / res)) // k * k)
    halo_px = max(1, int(round(halo_m / res)))
    out_h, out_w = spec["height"] // k, spec["width"] // k
    mosaic = np.full((out_h, out_w), np.nan, np.float32)
    fine_h, fine_w = spec["height"] // k * k, spec["width"] // k * k
    row_starts = list(range(0, fine_h, tile_px))
    col_starts = list(range(0, fine_w, tile_px))

    def window(r0: int, c0: int) -> dict:
        r1 = min(fine_h, r0 + tile_px)
        c1 = min(fine_w, c0 + tile_px)
        rr0, cc0 = max(0, r0 - halo_px), max(0, c0 - halo_px)
        rr1 = min(spec["height"], r1 + halo_px)
        cc1 = min(spec["width"], c1 + halo_px)
        ymin = spec["top"] - rr1 * res
        ymax = spec["top"] - rr0 * res
        xmin = spec["left"] + cc0 * res
        xmax = spec["left"] + cc1 * res
        return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "rr0": rr0, "cc0": cc0,
                "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
                "h": rr1 - rr0, "w": cc1 - cc0}

    n_col = len(col_starts)
    n_row = len(row_starts)
    tiles = [window(r0, c0) for r0 in row_starts for c0 in col_starts]
    # Один проход по LAS. Точка лежит в своём куске и в соседях, чьё поле до неё достаёт.
    reach = max(1, int(np.ceil(halo_px / tile_px)))
    import tempfile

    tmp_dir = os.environ.get("EXPDS_TMP") or None
    if tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
    use_all = not has_class(las_path, ground_class)
    if use_all:
        log.info("%s: класса %s нет, мозаика по всем точкам", Path(las_path).name, ground_class)
    tmp = tempfile.TemporaryDirectory(prefix="dem-mosaic-", dir=tmp_dir)
    try:
        bins = [(Path(tmp.name) / f"{i}.f64").open("wb") for i in range(len(tiles))]
        try:
            with laspy.open(str(las_path)) as f:
                for pts in f.chunk_iterator(1_000_000):
                    y_all = np.asarray(pts.y, np.float64)
                    if use_all:
                        ground = np.ones(np.asarray(pts.classification).shape, dtype=bool)
                    else:
                        ground = np.asarray(pts.classification) == ground_class
                    if not ground.any():
                        continue
                    x = np.asarray(pts.x, np.float64)[ground]
                    y = y_all[ground]
                    z = np.asarray(pts.z, np.float64)[ground]
                    col = np.floor((x - spec["left"]) / res).astype(np.int64)
                    row = np.floor((spec["top"] - y) / res).astype(np.int64)
                    ri = np.clip(row // tile_px, 0, max(n_row - 1, 0))
                    ci = np.clip(col // tile_px, 0, max(n_col - 1, 0))
                    for di in range(-reach, reach + 1):
                        for dj in range(-reach, reach + 1):
                            ti = ri + di
                            tj = ci + dj
                            ok = (ti >= 0) & (ti < n_row) & (tj >= 0) & (tj < n_col)
                            if not ok.any():
                                continue
                            ids = ti[ok] * n_col + tj[ok]
                            order = np.argsort(ids, kind="mergesort")
                            ids = ids[order]
                            xo, yo, zo = x[ok][order], y[ok][order], z[ok][order]
                            cuts = np.flatnonzero(np.diff(ids)) + 1
                            starts = np.r_[0, cuts]
                            for sl, tile_i in zip(np.split(np.arange(ids.size), cuts), ids[starts]):
                                t = tiles[int(tile_i)]
                                sel = ((xo[sl] >= t["xmin"]) & (xo[sl] <= t["xmax"])
                                       & (yo[sl] >= t["ymin"]) & (yo[sl] <= t["ymax"]))
                                if not sel.any():
                                    continue
                                blob = np.column_stack((xo[sl][sel], yo[sl][sel], zo[sl][sel]))
                                bins[int(tile_i)].write(np.ascontiguousarray(blob, np.float64).tobytes())
        finally:
            for fh in bins:
                fh.close()
        for i, t in enumerate(tiles):
            raw = np.fromfile(Path(tmp.name) / f"{i}.f64", dtype=np.float64)
            if raw.size:
                xyz = raw.reshape(-1, 3)
                c = np.floor((xyz[:, 0] - t["xmin"]) / res).astype(np.int64)
                r = np.floor((t["ymax"] - xyz[:, 1]) / res).astype(np.int64)
                inside = (c >= 0) & (r >= 0) & (c < t["w"]) & (r < t["h"])
                if inside.any():
                    flat = r[inside] * t["w"] + c[inside]
                    sums = np.bincount(flat, weights=xyz[inside, 2], minlength=t["h"] * t["w"])
                    counts = np.bincount(flat, minlength=t["h"] * t["w"])
                    local = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan).astype(np.float32)
                else:
                    local = np.full(t["h"] * t["w"], np.nan, np.float32)
            else:
                local = np.full(t["h"] * t["w"], np.nan, np.float32)
            local = fill_interior_holes(local.reshape(t["h"], t["w"]), method=fill)
            interior = local[t["r0"] - t["rr0"]:t["r1"] - t["rr0"], t["c0"] - t["cc0"]:t["c1"] - t["cc0"]]
            block = downsample_mean(interior, k)
            mosaic[t["r0"] // k:t["r0"] // k + block.shape[0],
                    t["c0"] // k:t["c0"] // k + block.shape[1]] = block
    finally:
        tmp.cleanup()
    if crs is None:
        from .crs import las_crs
        crs = las_crs(las_path)
    info = dict(spec, res=work_res, factor=1, tiled=True, fine_res=res,
                width=out_w, height=out_h, used_all=use_all, las_path=str(las_path), ground_class=ground_class)
    return Dem(dem=mosaic, left=spec["left"], top=spec["top"], res=work_res, crs=crs, info=info)


# Готовые сетки остаются до конца прогона: канал σ читает тот же DEM, что и основной лидар.
# 8 ГиБ хватает на копию публичной выборки (около 1 ГиБ сеток). Сверх этого старые сетки
# забываются и при повторном запросе строятся заново, числа те же.
_READY_BUDGET = int(float(os.environ.get("EXPDS_DEM_CACHE_GIB", "8")) * 1024 ** 3)


class DemPrefetch:
    """DEM следующих LAS строятся в фоновых потоках, пока текущий участок считается.

    Сетка, TIN и заливка — Rust без GIL, так что потоки действительно параллельны
    основному: тот в это время ждёт видеокарту и собирает полигоны. Чтение
    холодного файла с диска тоже уходит в фон. Триангуляция Делоне идёт в одном
    потоке, поэтому depth облаков триангулируются одновременно на разных ядрах.
    Уже запущенную сборку не отменяем: cancel не останавливает поток, а следующий
    запрос того же шага построил бы ту же сетку ещё раз. Готовая сетка хранится
    и отдаётся повторно, в том числе каналу σ после оптики.
    """

    def __init__(self, order: list[tuple[Path, str | None]], depth: int = 2):
        from concurrent.futures import ThreadPoolExecutor

        self.depth = max(1, int(depth))
        self._ahead = {str(p): order[i + 1:i + 1 + self.depth] for i, (p, _) in enumerate(order)}
        self._pool = ThreadPoolExecutor(max_workers=self.depth, thread_name_prefix="dem")
        self._jobs: dict[tuple, object] = {}
        self._ready: dict[tuple, Dem] = {}

    def _submit(self, las: Path, res: float, fill: str, crs: str | None) -> tuple:
        key = (str(las), float(res), fill, crs)
        if key not in self._ready and key not in self._jobs:
            self._jobs[key] = self._pool.submit(build_dem, las, res=res, fill=fill, crs=crs)
        return key

    def _queue_ahead(self, las: Path, res: float, fill: str) -> None:
        for path, crs in self._ahead.get(str(las), []):
            self._submit(path, res, fill, crs)

    def _remember(self, key: tuple, dem: Dem) -> Dem:
        self._ready.pop(key, None)
        self._ready[key] = dem
        while len(self._ready) > 1 and sum(item.dem.nbytes for item in self._ready.values()) > _READY_BUDGET:
            self._ready.pop(next(iter(self._ready)))
        return self._ready.get(key, dem)

    def get(self, las: Path, res: float, fill: str, crs: str | None) -> Dem:
        """DEM этого LAS; следующие depth по порядку ставятся в очередь сразу за ним."""
        key = (str(las), float(res), fill, crs)
        if key in self._ready:
            self._queue_ahead(las, res, fill)
            dem = self._ready.pop(key)
            return self._remember(key, dem)
        key = self._submit(las, res, fill, crs)
        keep = {key} | {self._submit(path, res, fill, ahead_crs)
                        for path, ahead_crs in self._ahead.get(str(las), [])}
        for stale in [k for k in self._jobs if k not in keep]:
            fut = self._jobs[stale]
            if fut.running():
                continue
            self._jobs.pop(stale)
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self._remember(stale, fut.result())
            elif not fut.done():
                fut.cancel()
        dem = self._jobs.pop(key).result()
        return self._remember(key, dem)

    def close(self) -> None:
        for fut in self._jobs.values():
            fut.cancel()
        self._jobs.clear()
        self._ready.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


def dem_params(meta: dict | None = None, cfg: dict | None = None) -> dict:
    """Параметры DEM: из чекпойнта модели li (как при обучении), иначе из [lidar] конфига, иначе MODEL_DEM."""
    if meta is not None:
        if meta.get("dem_res") is None:
            return dict(LEGACY_DEM)
        return {"res": float(meta["dem_res"]), "fill": meta.get("dem_fill") or MODEL_DEM["fill"]}
    lid = (cfg or {}).get("lidar", {})
    return {"res": float(lid.get("dem_res", MODEL_DEM["res"])), "fill": lid.get("dem_fill", MODEL_DEM["fill"])}


HILLSHADE_STYLES = ("vendor", "li_match")


def _hillshade_li_match(filled: np.ndarray, res: float) -> np.ndarray:
    """CI2-8 (W2 T3, notes/w2/ci2_8/SUMMARY.md): отмывка ближе к Li_g организаторов —
    Horn 3×3 по DEM, сглаженному гауссом σ=1 px; ESRI 315/45/z1. MAE к Li_g 2,4–4,6 против 3,1–6,1 у vendor."""
    import math
    from scipy import ndimage as ndi

    z = ndi.gaussian_filter(np.asarray(filled, np.float32), 1.0)
    k = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32) / (8.0 * res)
    gx = -ndi.correlate(z, k, mode="nearest")
    gy = ndi.correlate(z, k.T, mode="nearest")
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, -gy)
    zen, az = math.radians(45.0), math.radians(315.0)
    shade = 255.0 * (math.cos(zen) * np.cos(slope) + math.sin(zen) * np.sin(slope) * np.cos(az - aspect))
    return np.rint(np.clip(shade, 0.0, 255.0)).astype(np.uint8)


def hillshade_for(cfg: dict | None):
    """hillshade_set с lidar.hillshade_style из конфига (по умолчанию "vendor" — как было)."""
    style = str(((cfg or {}).get("lidar", {}) or {}).get("hillshade_style", "vendor"))
    if style not in HILLSHADE_STYLES:
        raise ValueError(f"lidar.hillshade_style={style!r}, ожидается одно из {HILLSHADE_STYLES}")
    if style == "vendor":
        return hillshade_set
    import functools
    return functools.partial(hillshade_set, style=style)


def hillshade_set(d: Dem, tags=TAGS, cycles_c: float = DEFAULTS["cycles_c"],
                  cycles_ch: float | None = None, style: str = "vendor") -> dict[str, np.ndarray]:
    """Отмывки по DEM: g, i — H×W uint8 (nodata 255); c, ch — 3×H×W uint8 (nodata 0)."""
    valid = np.isfinite(d.dem)
    if not valid.any():
        raise ValueError("DEM пуст")
    z_min = float(np.nanmin(d.dem))
    z_max = float(np.nanmax(d.dem))
    z_range = max(z_max - z_min, 1e-9)
    filled = np.where(valid, d.dem, np.float32(z_min)).astype(np.float32, copy=False)
    out: dict[str, np.ndarray] = {}
    if "g" in tags or "i" in tags:
        shade = (_hillshade_li_match(filled, d.res) if style == "li_match"
                 else accel.hillshade(filled, d.res, d.res))
        if "g" in tags:
            g = shade.copy()
            g[~valid] = 255
            out["g"] = g
        if "i" in tags:
            # Li организаторов: g + i = 253 (CI2-8), у vendor 255
            i = np.clip((253 if style == "li_match" else 255) - shade.astype(np.int16), 0, 255).astype(np.uint8)
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


def intensity_grid(las_path: str | Path, left: float, top: float, step: float, shape: tuple[int, int],
                   ground_class: int = DEFAULTS["ground_class"], chunk: int = 2_000_000) -> np.ndarray:
    """Средняя интенсивность точек земли в ячейках сетки (север вверху), NaN — пусто.

    Без точек класса земли берутся все точки, как в мозаике. Сетка — та же, что у DEM
    на рабочем шаге: ячейка (r, c) покрывает x ∈ [left + c·step, …), y ∈ (top − (r+1)·step, …].
    """
    import laspy

    h, w = shape
    sums = np.zeros(h * w, np.float64)
    cnts = np.zeros(h * w, np.float64)
    use_all = not has_class(las_path, ground_class)
    with laspy.open(str(las_path)) as f:
        for pts in f.chunk_iterator(chunk):
            keep = slice(None) if use_all else np.asarray(pts.classification) == ground_class
            x = np.asarray(pts.x)[keep]
            y = np.asarray(pts.y)[keep]
            v = np.asarray(pts.intensity, np.float64)[keep]
            c = np.floor((x - left) / step)
            r = np.floor((top - y) / step)
            ok = (c >= 0) & (r >= 0) & (c < w) & (r < h)
            flat = r[ok].astype(np.int64) * w + c[ok].astype(np.int64)
            sums += np.bincount(flat, weights=v[ok], minlength=h * w)
            cnts += np.bincount(flat, minlength=h * w)
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan).astype(np.float32).reshape(h, w)


def intensity_unit(d: Dem, step: float, shape: tuple[int, int]) -> np.ndarray:
    """Канал int в долях 0…1: перцентили INTENSITY_PCT облака, пустоты — среднее окна.

    У DEM без пути к LAS (собран не build_dem) канал постоянный 0,5 с предупреждением:
    модель с int на таком входе работает, но без этого канала.
    """
    path = d.info.get("las_path")
    if not path:
        log.warning("канал int: у DEM нет пути к LAS, канал постоянный")
        return np.full(shape, 0.5, np.float32)
    g = intensity_grid(path, d.left, d.top, step, shape, int(d.info.get("ground_class", DEFAULTS["ground_class"])))
    ok = np.isfinite(g)
    if not ok.any():
        log.warning("канал int: в %s нет точек на сетке, канал постоянный", Path(path).name)
        return np.full(shape, 0.5, np.float32)
    lo, hi = np.percentile(g[ok], INTENSITY_PCT)
    unit = np.clip((g - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)
    if not ok.all():
        near = accel.box_mean(unit, max(1, round(INTENSITY_FILL_M / step)))
        unit = np.where(ok, unit, np.where(np.isfinite(near), near, 0.5)).astype(np.float32)
    return unit


def _smooth_valid(z: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    """Гаусс только по пикселям съёмки (нормированная свёртка), float64: край облака не тянет к среднему."""
    from scipy.ndimage import gaussian_filter

    w = gaussian_filter(valid.astype(np.float64), sigma_px, mode="constant")
    s = gaussian_filter(np.where(valid, z, 0.0).astype(np.float64), sigma_px, mode="constant")
    return np.where(w > 1e-6, s / np.maximum(w, 1e-6), 0.0)


def _roughness(z: np.ndarray, valid: np.ndarray, step: float) -> np.ndarray:
    """RMS высоты над гладкой поверхностью, м. Плоскость любого уклона даёт ноль, азимута нет.

    Остаток после гаусса rough_hp_m, RMS в окне rough_win_m. СКО окна 3×3 без вычета
    поверхности на шаге 0,5 м — это тот же уклон, а шкала 1 м занимала 4–21 уровень из 255.
    """
    hp = np.where(valid, z - _smooth_valid(z, valid, RELIEF["rough_hp_m"] / step), 0.0)
    return np.sqrt(np.clip(_smooth_valid(hp * hp, valid, RELIEF["rough_win_m"] / step), 0, None))


def _curvature(z: np.ndarray, valid: np.ndarray, step: float) -> np.ndarray:
    """Лапласиан поверхности, сглаженной гауссом curv_sigma_m, 1/м. Купол меньше нуля, яма больше.

    Лапласиан 3×3 сырой ЦМР на шаге 0,5 м — это шум: при шуме высоты 1–5 см 78–96 %
    фона упирались в край шкалы. Гаусс 1,5 м держит курган радиусом от 3 м.
    """
    p = np.pad(_smooth_valid(z, valid, RELIEF["curv_sigma_m"] / step), 1, mode="edge")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]) / (step * step)


def relief_channels(d: Dem, res: float, tags=RELIEF_TAGS):
    """Каналы модели li из DEM на рабочем шаге: C×H×W uint8 и привязка (север вверху).

    Одна и та же функция строит кэш тайлов и вход модели на инференсе: в
    валидации готовых карт Li нет, а карты Li трейна построены другим
    инструментом и не повторяются по LAS (expds.lidar check). Значения 1…255,
    0 — вне контура съёмки.
    lrm_s, lrm_l — локальный рельеф (высота − среднее окна радиусом 8 и 30 м,
    под курганы лесной и степной зон); lrm_xl — то же окно 100 м под городища,
    в сданных весах курганов его нет; slope — уклон; g — серая отмывка;
    opns, svf — положительная открытость и фактор вида на небо (насыпь и ров);
    int — интенсивность отражения точек земли (intensity_unit), читается из LAS.
    dev_s, dev_l, dev_xl — то же окно, но делённое на σ окрестности и обрезанное
    к ±3: низкая насыпь в ровном поле не тонет, а в шумном лесу не сливается
    с клипом в метрах. Сданные каналы этот тег не использует.
    rough — RMS высоты над гладкой поверхностью (корневая шкала до 0,25 м),
    curv — лапласиан поверхности после гаусса 1,5 м (±0,1 1/м). Оба не зависят
    от азимута солнца. В сданных весах их нет: канал появляется, только если кэш
    или инференс попросили этот тег.
    Только g зависит от направления света: без него допустимы повороты.
    """
    from rasterio.transform import from_origin

    from .candidates import to_work_grid

    rs = accel._rs
    tags = tuple(tags)
    fused = [t for t in tags if t != "int"]
    if rs is not None and hasattr(rs, "relief_channels") and fused and set(fused) <= FUSED_TAGS:
        try:
            img, step = rs.relief_channels(
                np.ascontiguousarray(d.dem, np.float32), float(d.res), float(res), fused,
                float(RELIEF["lrm_s_m"]), float(RELIEF["lrm_s_clip_m"]),
                float(RELIEF["lrm_l_m"]), float(RELIEF["lrm_l_clip_m"]),
                float(RELIEF["slope_max_deg"]))
            step = float(step)
            if "int" in tags:
                unit = intensity_unit(d, step, img.shape[1:])
                ch = np.where(img[0] > 0, np.rint(1 + unit * 254), 0).astype(np.uint8)
                img = np.insert(img, tags.index("int"), ch, axis=0)
            return img, from_origin(d.left, d.top, step, step)
        except Exception as exc:
            log.debug("каналы li на Rust не собраны (%s), считаю в NumPy", exc)
    z, step = to_work_grid(d.dem, d.res, res)
    valid = np.isfinite(z)
    # Пустые ячейки перед уклоном и отмывкой заменяются средней высотой. У края дырки
    # это ложный обрыв, и он уже в сданных весах: кэш тайлов и инференс идут сюда же.
    # Отсев slope_gate эту заливку не повторяет и сеть не кормит.
    mean = np.float32(np.nanmean(z, dtype=np.float64)) if valid.any() else np.float32(0)
    filled = np.where(valid, z, mean).astype(np.float32, copy=False)
    out = []
    opn_svf = None
    for t in tags:
        if t in ("lrm_s", "lrm_l", "lrm_xl"):
            radius = max(1, round(RELIEF[f"{t}_m"] / step))
            lr = accel.local_relief(np.where(valid, z, np.nan).astype(np.float32, copy=False), radius)
            unit = np.clip(np.nan_to_num(lr, nan=0.0) / RELIEF[f"{t}_clip_m"], -1, 1) * 0.5 + 0.5
        elif t in ("dev_s", "dev_l", "dev_xl"):
            radius = max(1, round(RELIEF[f"{t}_m"] / step))
            dev = accel.local_dev(np.where(valid, z, np.nan).astype(np.float32, copy=False), radius,
                                  float(RELIEF["dev_std_floor_m"]))
            unit = np.clip(np.nan_to_num(dev, nan=0.0) / RELIEF["dev_clip"], -1, 1) * 0.5 + 0.5
        elif t == "slope":
            unit = np.clip(slope_deg(filled, step) / RELIEF["slope_max_deg"], 0, 1)
        elif t == "g":
            unit = accel.hillshade(filled, step, step).astype(np.float32) / 255.0
        elif t in ("opns", "svf"):
            if opn_svf is None:
                opn_svf = accel.openness_svf(np.where(valid, z, np.nan).astype(np.float32), step,
                                             max(1, round(RELIEF["opns_r_m"] / step)), int(RELIEF["opns_dirs"]))
            v = opn_svf[0] if t == "opns" else opn_svf[1]
            lo, hi = RELIEF[f"{t}_range"]
            unit = np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo), 0, 1)
        elif t == "int":
            unit = intensity_unit(d, step, z.shape)
        elif t == "rough":
            unit = np.sqrt(np.clip(_roughness(z, valid, step) / RELIEF["rough_clip_m"], 0, 1))
        elif t == "curv":
            unit = np.clip(_curvature(z, valid, step) / RELIEF["curv_clip"], -1, 1) * 0.5 + 0.5
        elif t == "prod":
            slope_u = np.clip(slope_deg(filled, step) / RELIEF["slope_max_deg"], 0, 1)
            radius = max(1, round(RELIEF["lrm_s_m"] / step))
            lr = accel.local_relief(np.where(valid, z, np.nan).astype(np.float32, copy=False), radius)
            lr_u = np.clip(np.nan_to_num(lr, nan=0.0) / RELIEF["lrm_s_clip_m"], -1, 1) * 0.5 + 0.5
            unit = np.clip(slope_u * lr_u, 0, 1)
        elif t == "ndi":
            slope_u = np.clip(slope_deg(filled, step) / RELIEF["slope_max_deg"], 0, 1)
            inten = intensity_unit(d, step, z.shape)
            unit = np.clip((slope_u - inten) / (slope_u + inten + 1e-6), -1, 1) * 0.5 + 0.5
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
