"""Признаки облака точек (cf_*) вокруг кандидатов — для реранкера RR1s-cf (J2). По умолчанию выключено.

Чистая реализация того же расчёта, что в обучении (w1/cf, ядро koz3-w5-cloudfeat, bot-W1):
только laspy + numpy + pyproj, без сети и S3. Читает LAS участка потоком (по 5 млн точек).

    [w5_cloudfeat]
    enabled = false      # включать вместе с реранкером, обученным на X4 (X3 + 10 cf)
    reserve_s = 120      # не считать, если до дедлайна меньше (участок → NaN, как без LAS)

Кандидат: центр и габариты — в UTM-зоне по центроиду объекта (как официальная метрика
и export_cands.py: центр → EPSG:3857, округление 0,01 м; w, h — bbox в метрах, 0,01 м).
Дальше центр → CRS LAS (UTM.json участка). Сетка 0,5 м от floor(header.mins), диск
r = clip(round(max(w, h)/2 / 0,5)·0,5, 2, 15) м и кольцо до r + 8 м; точки классов 7 и 18
выброшены; земля — классы 2 и 8.

Признаки (NAMES): плотность земли в диске/кольце, плотность всех точек в диске, доля
не первых отражений в диске/кольце, σ высоты земли в диске/кольце (NaN при ≤ 2 точках),
доля пустых по земле ячеек в диске/кольце, отношение плотностей земли диск/кольцо.
Значения округлены до 5 значащих цифр (в обучении CSV писался с '%.5g') и приведены к float32.
Нет LAS, нет CRS, пустая ROI или ошибка чтения → вся строка NaN (так же было в обучении:
LightGBM обрабатывает NaN штатно).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

R = 0.5
RING = 8.0
RMIN, RMAX = 2.0, 15.0
CHUNK = 5_000_000
NAMES = ("cf_gd_in", "cf_gd_ctx", "cf_pd_in", "cf_nf_in", "cf_nf_ctx", "cf_zs_in", "cf_zs_ctx",
         "cf_hole_in", "cf_hole_ctx", "cf_gd_ratio")
KEY = "cf"          # ключ объекта пайплайна: np.ndarray(10, float32) в порядке NAMES; в GeoJSON не пишется

_OFF: dict[float, tuple] = {}


def offsets(rin: float):
    """Смещения ячеек в радиусе rin + RING и маска «в диске» (d ≤ rin)."""
    if rin not in _OFF:
        n = int(math.ceil((rin + RING) / R))
        dy, dx = np.mgrid[-n:n + 1, -n:n + 1]
        d = R * np.hypot(dx, dy).ravel()
        k = d <= rin + RING
        _OFF[rin] = (dx.ravel()[k], dy.ravel()[k], d[k] <= rin)
    return _OFF[rin]


def nan_block(n: int) -> np.ndarray:
    return np.full((n, len(NAMES)), np.nan, np.float32)


def round5(F: np.ndarray) -> np.ndarray:
    """Как в обучении: float(f'{v:.5g}') → float32; не конечные → NaN."""
    out = np.full(F.shape, np.nan, np.float32)
    flat, o = F.ravel(), out.ravel()
    for i in np.flatnonzero(np.isfinite(flat)):
        o[i] = float(f"{float(flat[i]):.5g}")
    return out


def cand_boxes(geoms) -> np.ndarray:
    """Геометрии EPSG:3857 → (n, 4): cx, cy (EPSG:3857), w, h (м, UTM-зона по центроиду), как export_cands.py."""
    import shapely
    from pyproj import Transformer

    geoms = np.asarray(list(geoms), dtype=object)
    out = np.full((len(geoms), 4), np.nan)
    if not len(geoms):
        return out
    cen = shapely.centroid(geoms)
    lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(shapely.get_x(cen), shapely.get_y(cen))
    zones = np.clip(((np.asarray(lon) + 180.0) // 6).astype(np.int32) + 1, 1, 60)
    epsgs = np.where(np.asarray(lat) >= 0, 32600, 32700) + zones
    for e in np.unique(epsgs):
        ii = np.flatnonzero(epsgs == e)
        to_utm = Transformer.from_crs(3857, int(e), always_xy=True)
        back = Transformer.from_crs(int(e), 3857, always_xy=True)
        ug = shapely.transform(geoms[ii], lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])))
        uc = shapely.centroid(ug)
        bx, by = back.transform(shapely.get_x(uc), shapely.get_y(uc))
        b = shapely.bounds(ug)
        out[ii, 0], out[ii, 1] = np.round(bx, 2), np.round(by, 2)
        out[ii, 2], out[ii, 3] = np.round(b[:, 2] - b[:, 0], 2), np.round(b[:, 3] - b[:, 1], 2)
    return out


def _epsg_or_wkt(crs):
    s = str(crs)
    if s.upper().startswith("EPSG:") and s.split(":")[-1].isdigit():
        return int(s.split(":")[-1])
    return s


def compute(las_paths, crs, boxes: np.ndarray, rounded: bool = True, stats: dict | None = None) -> np.ndarray:
    """(n, 10) float32. las_paths — LAS участка (обычно один), crs — CRS LAS (UTM.json), boxes — cand_boxes()."""
    import laspy
    from pyproj import Transformer

    boxes = np.asarray(boxes, float).reshape(-1, 4)
    N = len(boxes)
    las_paths = [Path(p) for p in (las_paths or [])]
    if not N or not las_paths or crs is None:
        return nan_block(N)
    ok_box = np.isfinite(boxes).all(1)
    ux = np.full(N, np.nan)
    uy = np.full(N, np.nan)
    if ok_box.any():
        tr = Transformer.from_crs(3857, _epsg_or_wkt(crs), always_xy=True)
        ux[ok_box], uy[ok_box] = tr.transform(boxes[ok_box, 0], boxes[ok_box, 1])
    ok_box &= np.isfinite(ux) & np.isfinite(uy)
    q_all = np.flatnonzero(ok_box)
    if not len(q_all):
        return nan_block(N)
    rin = np.clip(np.round(0.5 * np.maximum(boxes[:, 2], boxes[:, 3]) / R) * R, RMIN, RMAX)

    heads = []
    for p in las_paths:
        with laspy.open(str(p)) as fh:
            heads.append((np.asarray(fh.header.mins, float), np.asarray(fh.header.maxs, float)))
    mins = np.min([h[0] for h in heads], 0)
    maxs = np.max([h[1] for h in heads], 0)
    x0, y0 = math.floor(mins[0]), math.floor(mins[1])
    W = int((maxs[0] - x0) / R) + 2
    H = int((maxs[1] - y0) / R) + 2
    zr = float(mins[2])
    c0 = np.zeros(N, np.int64)
    r0 = np.zeros(N, np.int64)
    c0[q_all] = np.floor((ux[q_all] - x0) / R).astype(np.int64)
    r0[q_all] = np.floor((uy[q_all] - y0) / R).astype(np.int64)

    # ROI — объединение ячеек всех дисков и колец (линейные номера, по возрастанию)
    U = np.zeros(0, np.int64)
    for rv in np.unique(rin[q_all]):
        dx, dy, _ = offsets(float(rv))
        jj = q_all[rin[q_all] == rv]
        bs = max(1, 2_000_000 // len(dx))
        for b in range(0, len(jj), bs):
            q = jj[b:b + bs]
            cc = c0[q, None] + dx[None]
            rr = r0[q, None] + dy[None]
            ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
            U = np.union1d(U, (rr * W + cc)[ok])
    n = len(U)
    if not n:
        return nan_block(N)
    na = np.zeros(n)
    ng = np.zeros(n)
    nf = np.zeros(n)
    sz = np.zeros(n)
    sz2 = np.zeros(n)
    npts = 0
    for p in las_paths:
        with laspy.open(str(p)) as fh:
            for pt in fh.chunk_iterator(CHUNK):
                x = np.asarray(pt.x)
                y = np.asarray(pt.y)
                cl = np.asarray(pt.classification)
                npts += len(x)
                lin = ((y - y0) / R).astype(np.int64) * W + ((x - x0) / R).astype(np.int64)
                i = np.searchsorted(U, lin)
                i[i >= n] = n - 1
                m = (U[i] == lin) & (cl != 7) & (cl != 18)
                if not m.any():
                    continue
                i = i[m]
                cl = cl[m]
                z = np.asarray(pt.z)[m] - zr
                rn = np.asarray(pt.return_number)[m]
                g = (cl == 2) | (cl == 8)
                na += np.bincount(i, minlength=n)
                nf += np.bincount(i, weights=(rn > 1).astype(float), minlength=n)
                ng += np.bincount(i[g], minlength=n)
                sz += np.bincount(i[g], weights=z[g], minlength=n)
                sz2 += np.bincount(i[g], weights=z[g] ** 2, minlength=n)

    def agg(sel, cnt):
        def v(a):
            return np.where(sel >= 0, a[np.maximum(sel, 0)], 0.0) * cnt
        A, G, NF, S, S2 = v(na).sum(1), v(ng).sum(1), v(nf).sum(1), v(sz).sum(1), v(sz2).sum(1)
        cells = cnt.sum(1)
        hole = (((v(ng) == 0) & cnt).sum(1)) / np.maximum(cells, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(G > 2, S2 / np.maximum(G, 1) - (S / np.maximum(G, 1)) ** 2, np.nan)
            return (G / (cells * R * R), A / (cells * R * R), np.where(A > 0, NF / np.maximum(A, 1), np.nan),
                    np.sqrt(np.maximum(var, 0)), hole)

    F = nan_block(N)
    for rv in np.unique(rin[q_all]):
        dx, dy, inn = offsets(float(rv))
        jj = q_all[rin[q_all] == rv]
        bs = max(1, 3_000_000 // len(dx))
        for b in range(0, len(jj), bs):
            q = jj[b:b + bs]
            cc = c0[q, None] + dx[None]
            rr = r0[q, None] + dy[None]
            ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
            lin = rr * W + cc
            s = np.searchsorted(U, lin)
            s[s >= n] = 0
            s = np.where(ok & (U[s] == lin), s, -1)
            gi, pi, nfi, zi, hi = agg(s, np.broadcast_to(inn[None], s.shape))
            go, po, nfo, zo, ho = agg(s, np.broadcast_to(~inn[None], s.shape))
            F[q] = np.stack([gi, go, pi, nfi, nfo, zi, zo, hi, ho, gi / np.maximum(go, 1e-3)], 1)
    if stats is not None:
        stats.update(points=int(npts), roi_cells=int(n), cands=int(N))
    return round5(F) if rounded else F


def region_features(feats: list[dict], las_paths, crs) -> np.ndarray:
    """Признаки для объектов пайплайна одного участка (geom в EPSG:3857). Ошибка → NaN."""
    n = len(feats)
    if not n:
        return nan_block(0)
    if not las_paths or crs is None:
        return nan_block(n)
    try:
        return compute(las_paths, crs, cand_boxes([f["geom"] for f in feats]))
    except Exception:
        log.exception("w5_cloudfeat: расчёт не удался, признаки участка → NaN")
        return nan_block(n)


def annotate(feats: list[dict], regions: dict, cfg: dict | None = None, crs_for=None, guard=None) -> list[dict]:
    """Добавляет f[KEY] (10 × float32) каждому объекту. regions: имя → Region; без LAS → NaN.

    crs_for(reg) → CRS LAS (по умолчанию UTM.json участка, как в обучении). Порядок объектов не меняется.
    """
    spec = cfg or {}
    reserve = float(spec.get("reserve_s", 120))
    if crs_for is None:
        from .catalog import read_utm_json

        def crs_for(r):
            return read_utm_json(getattr(r, "utm_json", None))
    by_reg: dict[str, list[int]] = {}
    for k, f in enumerate(feats):
        by_reg.setdefault(f.get("region_name"), []).append(k)
    out = [dict(f) for f in feats]
    for name, idx in by_reg.items():
        reg = regions.get(name)
        las = list(getattr(reg, "las", None) or []) if reg is not None else []
        crs = None
        if las:
            try:
                crs = crs_for(reg)
            except Exception:
                log.exception("w5_cloudfeat: %s: CRS не прочитан", name)
        if las and guard is not None and not guard.ok(reserve):
            log.warning("w5_cloudfeat: %s: нет времени, признаки → NaN", name)
            las = []
        F = region_features([feats[k] for k in idx], las, crs)
        for k, row in zip(idx, F):
            out[k][KEY] = row
        if las:
            log.info("w5_cloudfeat: %s: %d объектов, конечных строк %d", name, len(idx), int(np.isfinite(F).any(1).sum()))
    return out


def matrix(feats: list[dict]) -> np.ndarray:
    """(n, 10) из f[KEY]; объекты без ключа → NaN."""
    F = nan_block(len(feats))
    for k, f in enumerate(feats):
        v = f.get(KEY)
        if v is not None:
            F[k] = v
    return F
