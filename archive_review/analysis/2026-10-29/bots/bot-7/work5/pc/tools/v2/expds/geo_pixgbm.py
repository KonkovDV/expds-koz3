"""Пиксельный LightGBM-источник кандидатов геофизики: object_poly (maggr/etgr/terfl) и finds_points (maggr).

Чтение карты — по мотивам expds/geo_finds_gbm.py (ветка bot3-ehimen, своя реализация), переписано:
два скаляра (среднее RGB и разность R−B цветной шкалы), полосами, float32.

Признаки пикселя (все размеры в метрах, без поворотов и отражений — знак N−S диполя сохраняется):
  z, медианный тренд 5 м и 15 м (медиана на сетке 0,5 м, билинейно назад), d5 = z−med5, d15 = z−med15;
  LoG σ 0,2/0,4/0,8/1,6 м и DoG (0,2−0,8, 0,4−1,6, 0,8−3,2) по d15;
  локальные mean/std/min/max d5 в окнах 0,6/1,5/3 м, перцентили 10/90 в окне 1,5 м (сетка 0,25 м);
  градиент gx, gy (знаковые), |∇| на 0,4 и 1,2 м;
  кольца 16 лучей на 0,5/1/2 м по сглаженному d5: среднее−центр, std, N−S, E−W, min R_k·R_{k+8}, размах,
  16 сырых значений кольца 1 м минус центр; расстояние до края съёмки; метод (mt).
Карта режется на окна CORE + PAD; признаки считаются в окне и берутся только в нужных пикселях.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

GSD = {"maggr": 0.1, "etgr": 0.05, "terfl": 0.3}
MT_ID = {"maggr": 0, "etgr": 1, "terfl": 2}
PAD_M = 6.0
CORE = 512
RINGS = (0.5, 1.0, 2.0)
LGBM = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
            feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=2.0,
            verbose=-1, num_threads=2)
ROUNDS = 300
T_PIX = 0.5
MIN_AREA_M2 = 0.1
POLE_R_M = 0.05
FIND_NMS_M = 0.75
FIND_FLOOR = 0.2
FIND_R_M = 0.5


# ───────────────────────────── выбор карт ─────────────────────────────

def raster_key(path: Path) -> str:
    """Ключ карты: имя без номера в начале и без «_серый»/«_цвет» (цвет_2 — своя карта)."""
    s = re.sub(r"^\d+_", "", unicodedata.normalize("NFC", Path(path).stem))
    return re.sub(r"_(серый|цвет)$", "", s)


def pick_rasters(files: list[Path]) -> list[Path]:
    """Одна карта на ключ: серая, если есть (серая и цветная копия — одна съёмка)."""
    by: dict[str, list[Path]] = {}
    for f in sorted(files):
        by.setdefault(raster_key(f), []).append(f)
    out = []
    for k, fs in by.items():
        gray = [f for f in fs if "серый" in f.stem]
        out.append((gray or fs)[0])
    return out


def label_key(path: Path) -> str:
    return re.sub(r"_(ObjectPoly|FindsPoints)$", "", unicodedata.normalize("NFC", Path(path).stem))


# ───────────────────────────── чтение карты ─────────────────────────────

def load_map(path, gsd: float, fallback_crs=None, boundary=None, noise=None) -> dict:
    import rasterio
    from rasterio import features
    from rasterio.windows import Window

    from .crs import WEB_MERCATOR, to_crs, usable_raster_crs

    with rasterio.open(path) as r:
        f = max(1, int(round(gsd / abs(r.transform.a))))
        H, W = max(1, r.height // f), max(1, r.width // f)
        n = r.count
        alpha = [i for i, ci in enumerate(r.colorinterp) if getattr(ci, "name", "") == "alpha"]
        if not alpha and n == 4:
            alpha = [3]
        lum = np.zeros((H, W), np.float32)
        dif = np.zeros((H, W), np.float32)
        valid = np.zeros((H, W), bool)
        step = max(f, (6_000_000 // max(r.width * n, 1)) // f * f)
        for y0 in range(0, H * f, step):
            h = min(step, H * f - y0)
            blk = r.read(window=Window(0, y0, W * f, h)).astype(np.float32)
            blk = blk.reshape(n, h // f, f, W, f).mean((2, 4))
            rows = slice(y0 // f, (y0 + h) // f)
            rgb = blk[:3] if n >= 3 else np.repeat(blk[:1], 3, 0)
            v = ~((rgb.min(0) >= 254) | (rgb.max(0) <= 1))
            if alpha:
                v &= blk[alpha[0]] > 0
            lum[rows], dif[rows], valid[rows] = rgb.mean(0), rgb[0] - rgb[2], v
            del blk, rgb
        T = r.transform @ r.transform.scale(r.width / W, r.height / H)
        crs = usable_raster_crs(r.crs.to_string() if r.crs else None, r.bounds, fallback_crs)
    up = abs(T.a) / gsd
    if up > 1.3:
        h0, w0 = lum.shape
        lum = ndi.zoom(lum, up, order=1).astype(np.float32)
        dif = ndi.zoom(dif, up, order=1).astype(np.float32)
        valid = ndi.zoom(valid.astype(np.uint8), up, order=0).astype(bool)
        h1 = min(lum.shape[0], dif.shape[0], valid.shape[0])
        w1 = min(lum.shape[1], dif.shape[1], valid.shape[1])
        lum, dif, valid = lum[:h1, :w1], dif[:h1, :w1], valid[:h1, :w1]
        T = T @ T.scale(w0 / w1, h0 / h1)
    nz = np.zeros(lum.shape, bool)
    if crs:
        if boundary is not None:
            g = to_crs(boundary, WEB_MERCATOR, crs)
            if not g.is_empty:
                valid &= features.rasterize([(g, 1)], out_shape=lum.shape, transform=T, all_touched=True).astype(bool)
        if noise is not None:
            g = to_crs(noise, WEB_MERCATOR, crs)
            if not g.is_empty:
                nz = features.rasterize([(g, 1)], out_shape=lum.shape, transform=T, all_touched=True).astype(bool)
    return dict(z=_robust(lum, valid), zd=_robust(dif, valid), valid=valid, noise=nz, T=T, crs=crs,
                gsd=float(abs(T.a)))


def _robust(x, valid):
    v = x[valid]
    if v.size < 10:
        return np.zeros_like(x, np.float32)
    med = np.median(v)
    q1, q3 = np.percentile(v, [25, 75])
    z = (x - med) / max(q3 - q1, 1e-3)
    z[~valid] = 0
    return np.clip(z, -20, 20).astype(np.float32)


def trends(z, valid, gsd):
    """Медианы 5 м и 15 м на сетке 0,5 м (в данных; дыры — 0 после нормировки)."""
    b = max(1, int(round(0.5 / gsd)))
    H, W = z.shape
    hb, wb = -(-H // b), -(-W // b)
    zp = np.zeros((hb * b, wb * b), np.float32); zp[:H, :W] = np.where(valid, z, 0)
    vp = np.zeros((hb * b, wb * b), np.float32); vp[:H, :W] = valid
    s = zp.reshape(hb, b, wb, b).sum((1, 3)); c = vp.reshape(hb, b, wb, b).sum((1, 3))
    zc = np.where(c > 0, s / np.maximum(c, 1), 0).astype(np.float32)
    out = []
    for w_m in (5.0, 15.0):
        k = max(3, int(round(w_m / (b * gsd))) | 1)
        out.append(ndi.median_filter(zc, size=k, mode="nearest"))
    return out, b


def _up(coarse, b, y0, x0, h, w):
    yy = (np.arange(y0, y0 + h, dtype=np.float32) + 0.5) / b - 0.5
    xx = (np.arange(x0, x0 + w, dtype=np.float32) + 0.5) / b - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return ndi.map_coordinates(coarse, [Y, X], order=1, mode="nearest").astype(np.float32)


# ───────────────────────────── признаки ─────────────────────────────

def features_at(zw, zdw, vw, m5, m15, gsd, mt_id, rr, cc) -> dict:
    """Признаки в пикселях (rr, cc) окна. Все входы — окна одного размера."""
    def px(m):
        return max(m / gsd, 0.5)
    inv = ~vw
    F = {}
    d5 = np.where(vw, zw - m5, 0).astype(np.float32)
    d15 = np.where(vw, zw - m15, 0).astype(np.float32)
    F["z"], F["d5"], F["d15"], F["trend"] = zw[rr, cc], d5[rr, cc], d15[rr, cc], (m5 - m15)[rr, cc]
    g = {}
    for s in (0.2, 0.4, 0.8, 1.6, 3.2):
        g[s] = ndi.gaussian_filter(d15, px(s))
        if s <= 1.6:
            F[f"g{s}"] = g[s][rr, cc]
            lg = -ndi.gaussian_laplace(d15, px(s)) * px(s) ** 2
            F[f"log{s}"] = lg[rr, cc]
            del lg
    for a, b in ((0.2, 0.8), (0.4, 1.6), (0.8, 3.2)):
        F[f"dog{a}_{b}"] = (g[a] - g[b])[rr, cc]
    gy = ndi.gaussian_filter(d15, px(0.4), order=(1, 0)) * px(0.4)
    gx = ndi.gaussian_filter(d15, px(0.4), order=(0, 1)) * px(0.4)
    F["gx0.4"], F["gy0.4"], F["gm0.4"] = gx[rr, cc], gy[rr, cc], np.hypot(gx, gy)[rr, cc]
    gy = ndi.gaussian_filter(d15, px(1.2), order=(1, 0)) * px(1.2)
    gx = ndi.gaussian_filter(d15, px(1.2), order=(0, 1)) * px(1.2)
    F["gm1.2"] = np.hypot(gx, gy)[rr, cc]
    del gx, gy, g
    for w in (0.6, 1.5, 3.0):
        k = max(3, int(round(w / gsd)) | 1)
        m1 = ndi.uniform_filter(d5, k); m2 = ndi.uniform_filter(d5 * d5, k)
        F[f"mean{w}"], F[f"std{w}"] = m1[rr, cc], np.sqrt(np.maximum(m2 - m1 * m1, 0))[rr, cc]
        F[f"max{w}"], F[f"min{w}"] = ndi.maximum_filter(d5, k)[rr, cc], ndi.minimum_filter(d5, k)[rr, cc]
        del m1, m2
    q = max(1, int(round(0.25 / gsd)))
    H, W = d5.shape
    hq, wq = H // q, W // q
    if hq >= 3 and wq >= 3:
        dc = d5[:hq * q, :wq * q].reshape(hq, q, wq, q).mean((1, 3))
        k = max(3, int(round(1.5 / (q * gsd))) | 1)
        ri, ci = np.minimum(rr // q, hq - 1), np.minimum(cc // q, wq - 1)
        F["p10_1.5"] = ndi.percentile_filter(dc, 10, size=k)[ri, ci]
        F["p90_1.5"] = ndi.percentile_filter(dc, 90, size=k)[ri, ci]
    else:
        F["p10_1.5"] = F["min1.5"]; F["p90_1.5"] = F["max1.5"]
    sm = ndi.gaussian_filter(d5, px(0.2))
    c0 = sm[rr, cc]
    ang = np.arange(16) * (2 * np.pi / 16)          # 0 — восток, против часовой; 4 — север (строки вверх)
    north, south = list(range(1, 8)), list(range(9, 16))
    east, west = [13, 14, 15, 0, 1, 2, 3], list(range(5, 12))
    for rad in RINGS:
        rp = rad / gsd
        ys = np.clip(np.round(rr[:, None] - rp * np.sin(ang)[None]).astype(int), 0, H - 1)
        xs = np.clip(np.round(cc[:, None] + rp * np.cos(ang)[None]).astype(int), 0, W - 1)
        R = sm[ys, xs]
        F[f"r{rad}_mc"] = R.mean(1) - c0
        F[f"r{rad}_std"] = R.std(1)
        F[f"r{rad}_ns"] = R[:, north].mean(1) - R[:, south].mean(1)
        F[f"r{rad}_ew"] = R[:, east].mean(1) - R[:, west].mean(1)
        F[f"r{rad}_opp"] = (R[:, :8] * R[:, 8:]).min(1)
        F[f"r{rad}_rng"] = R.max(1) - R.min(1)
        if rad == 1.0:
            for k in range(16):
                F[f"r1_{k:02d}"] = R[:, k] - c0
    del sm
    dd = np.where(vw, zdw, 0).astype(np.float32)
    F["zd"] = dd[rr, cc]
    F["zd_log0.4"] = (-ndi.gaussian_laplace(dd, px(0.4)) * px(0.4) ** 2)[rr, cc]
    m1 = ndi.uniform_filter(dd, max(3, int(round(1.5 / gsd)) | 1))
    F["zd_hp1.5"] = (dd - m1)[rr, cc]
    F["edge"] = np.minimum(ndi.distance_transform_edt(vw) * gsd, PAD_M)[rr, cc]
    F["mt"] = np.full(len(rr), float(mt_id))
    return {k: np.asarray(v, np.float32) for k, v in F.items()}


def windows(H, W, gsd):
    pad = int(np.ceil(PAD_M / gsd))
    for ty in range(0, H, CORE):
        for tx in range(0, W, CORE):
            yield ty, tx, (max(ty - pad, 0), max(tx - pad, 0), min(ty + CORE + pad, H), min(tx + CORE + pad, W))


def features_for(d: dict, mt: str, rows: np.ndarray, cols: np.ndarray):
    """Признаки в пикселях карты (rows, cols) окнами. → (names, X float32 в порядке входа)."""
    (c5, c15), b = trends(d["z"], d["valid"], d["gsd"])
    H, W = d["z"].shape
    X, names = None, None
    for ty, tx, (y0, x0, y1, x1) in windows(H, W, d["gsd"]):
        sel = np.flatnonzero((rows >= ty) & (rows < ty + CORE) & (cols >= tx) & (cols < tx + CORE))
        if not sel.size:
            continue
        h, w = y1 - y0, x1 - x0
        F = features_at(d["z"][y0:y1, x0:x1], d["zd"][y0:y1, x0:x1], d["valid"][y0:y1, x0:x1],
                        _up(c5, b, y0, x0, h, w), _up(c15, b, y0, x0, h, w), d["gsd"], MT_ID[mt],
                        rows[sel] - y0, cols[sel] - x0)
        if X is None:
            names = list(F)
            X = np.zeros((len(rows), len(names)), np.float32)
        X[sel] = np.column_stack([F[k] for k in names])
        del F
    return names, X


def prob_map(d: dict, mt: str, boosters: dict) -> dict:
    """Карты вероятностей {имя: H×W float32} окнами; вне съёмки и в шуме — 0."""
    (c5, c15), b = trends(d["z"], d["valid"], d["gsd"])
    H, W = d["z"].shape
    use = d["valid"] & ~d["noise"]
    out = {k: np.zeros((H, W), np.float32) for k in boosters}
    for ty, tx, (y0, x0, y1, x1) in windows(H, W, d["gsd"]):
        cy1, cx1 = min(ty + CORE, H), min(tx + CORE, W)
        rr, cc = np.nonzero(use[ty:cy1, tx:cx1])
        if not rr.size:
            continue
        rr, cc = rr + ty - y0, cc + tx - x0
        h, w = y1 - y0, x1 - x0
        F = features_at(d["z"][y0:y1, x0:x1], d["zd"][y0:y1, x0:x1], d["valid"][y0:y1, x0:x1],
                        _up(c5, b, y0, x0, h, w), _up(c15, b, y0, x0, h, w), d["gsd"], MT_ID[mt], rr, cc)
        for k, bst in boosters.items():
            X = np.column_stack([F[n] for n in bst.feature_name()])
            out[k][rr + y0, cc + x0] = bst.predict(X, num_threads=LGBM["num_threads"]).astype(np.float32)
            del X
        del F
    return out


# ───────────────────────────── метки ─────────────────────────────

def label_masks(d: dict, polys, points):
    """(метка object_poly, метка finds) — int8 H×W: 1, 0, −1 (игнор); None — нет файла разметки."""
    from rasterio import features
    from shapely.geometry import Point
    from .crs import WEB_MERCATOR, to_crs

    shape, T, crs, gsd = d["z"].shape, d["T"], d["crs"], d["gsd"]
    out = []
    for geoms, kind in ((polys, "poly"), (points, "pt")):
        if geoms is None:
            out.append(None)
            continue
        gs = [to_crs(g, WEB_MERCATOR, crs) for g in geoms]
        if kind == "pt":
            pos_g = [Point(g.centroid.x, g.centroid.y).buffer(0.3) for g in gs]
            ign_g = [Point(g.centroid.x, g.centroid.y).buffer(1.0) for g in gs]
            pos = features.rasterize([(g, 1) for g in pos_g], out_shape=shape, transform=T, all_touched=True).astype(bool) if gs else np.zeros(shape, bool)
            ign = features.rasterize([(g, 1) for g in ign_g], out_shape=shape, transform=T, all_touched=True).astype(bool) if gs else np.zeros(shape, bool)
        else:
            pos = features.rasterize([(g, 1) for g in gs if not g.is_empty], out_shape=shape, transform=T).astype(bool) if gs else np.zeros(shape, bool)
            ign = ndi.binary_dilation(pos, iterations=max(1, int(round(0.3 / gsd)))) if gs else pos
        y = np.zeros(shape, np.int8)
        y[ign] = -1
        y[pos] = 1
        out.append(y)
    return out


# ───────────────────────────── выход ─────────────────────────────

def poly_candidates(P, d, region, mt, source, t_pix=T_PIX):
    """Компоненты P ≥ t_pix → круг POLE_R_M в полюсе недоступности; conf — средняя P компоненты."""
    from rasterio import features as rf
    from shapely.geometry import shape as shp
    from shapely.ops import polylabel, unary_union
    from .raster import circle_3857

    gsd, T = d["gsd"], d["T"]
    m = ndi.binary_opening(P >= t_pix)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if not n:
        return []
    out = []
    min_px = MIN_AREA_M2 / gsd / gsd
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        sub = lab[sl] == i
        if sub.sum() < min_px:
            continue
        conf = float(P[sl][sub].mean())
        tt = T * T.translation(sl[1].start, sl[0].start)
        geoms = [shp(g) for g, v in rf.shapes(sub.astype(np.uint8), mask=sub, transform=tt) if v == 1]
        g = unary_union(geoms)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        try:
            pt = polylabel(g, tolerance=0.02)
        except Exception:
            pt = g.representative_point()
        out.append({"geom": circle_3857(pt.x, pt.y, POLE_R_M, d["crs"]), "class_name": "object_poly",
                    "confidence": conf, "region_name": region, "markup_type": mt, "source": source,
                    "original_crs": d["crs"], "threshold_group": f"{mt}_pix", "area_m2": float(g.area)})
    return out


def poly_components(P, d, region, mt, source, levels=(0.3, 0.5, 0.7), min_px=2):
    """Сырые компоненты P ≥ t (для каждого t из levels) — полные полигоны EPSG:3857 (для валидации bot-6).
    Свойства: score — средняя P компоненты, max_p, t_pix, area_m2 (в метрах CRS карты), pole_x/pole_y (EPSG:3857)."""
    from shapely.geometry import Point
    from shapely.ops import polylabel
    from .raster import mask_to_polygons, to_crs, WEB_MERCATOR

    out = []
    for t in levels:
        m = ndi.binary_opening(P >= t)
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        if not n:
            continue
        idx = np.arange(1, n + 1)
        cnt = ndi.sum(np.ones_like(P), lab, idx)
        mean = ndi.mean(P, lab, idx)
        mxp = ndi.maximum(P, lab, idx)
        keep = np.zeros(n + 1, np.int32)
        keep[idx[cnt >= min_px]] = idx[cnt >= min_px]
        lab = keep[lab]
        native = mask_to_polygons(lab, d["T"], None)
        for g, v in native:
            if g.is_empty or not g.is_valid:
                g = g.buffer(0)
                if g.is_empty:
                    continue
                if g.geom_type == "MultiPolygon":
                    g = max(g.geoms, key=lambda x: x.area)
            try:
                pt = polylabel(g, tolerance=0.02)
            except Exception:
                pt = g.representative_point()
            g3, p3 = (to_crs(g, d["crs"], WEB_MERCATOR), to_crs(pt, d["crs"], WEB_MERCATOR)) if d["crs"] else (g, pt)
            out.append(dict(geom=g3, score=float(mean[v - 1]), max_p=float(mxp[v - 1]), t_pix=float(t),
                            area_m2=float(g.area), pole_x=float(p3.x), pole_y=float(p3.y),
                            region=region, markup_type=mt, raster=source))
    return out


def finds_candidates(P, d, region, mt, source):
    from scipy.spatial import cKDTree
    from .raster import circle_3857

    gsd, T = d["gsd"], d["T"]
    r = max(1, int(round(FIND_NMS_M / gsd)))
    mx = ndi.maximum_filter(P, size=2 * r + 1)
    rr, cc = np.nonzero((P == mx) & (P >= FIND_FLOOR))
    if not rr.size:
        return []
    o = np.argsort(-P[rr, cc], kind="stable")
    rr, cc = rr[o], cc[o]
    keep = np.ones(len(rr), bool)
    tree = cKDTree(np.c_[rr, cc])
    for i in range(len(rr)):
        if keep[i]:
            for j in tree.query_ball_point((rr[i], cc[i]), r):
                if j > i:
                    keep[j] = False
    out = []
    H, W = P.shape
    for y, x in zip(rr[keep], cc[keep]):
        y0, y1, x0, x1 = max(y - 1, 0), min(y + 2, H), max(x - 1, 0), min(x + 2, W)
        w = P[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        fy, fx = float((w * yy).sum() / w.sum()), float((w * xx).sum() / w.sum())
        X = T.c + (fx + 0.5) * T.a + (fy + 0.5) * T.b
        Y = T.f + (fx + 0.5) * T.d + (fy + 0.5) * T.e
        out.append({"geom": circle_3857(X, Y, FIND_R_M, d["crs"]), "class_name": "finds_points",
                    "confidence": float(P[y, x]), "region_name": region, "markup_type": mt, "source": source,
                    "original_crs": d["crs"], "threshold_group": f"{mt}_pix"})
    return out


def region_feats(reg, cfg, model_dir, stop=None) -> list[dict]:
    """Опциональный источник пайплайна: [geophys.pixgbm] enabled=true, model_dir (poly.txt, finds.txt).

    stop (bot-7) — функция без аргументов; True перед любым растром — участок целиком без pix (пустой
    список, классика остаётся): потолок времени pix или общий бюджет. Нет файлов моделей — предупреждение и пустой список.
    """
    import lightgbm as lgb
    from . import geophys
    from .pipeline import region_crs

    md = Path(model_dir)
    crs_region = region_crs(reg, cfg)
    geo_cfg = cfg.get("geophys", {})
    bst = {k: lgb.Booster(model_file=str(md / f"{k}.txt")) for k in ("poly", "finds") if (md / f"{k}.txt").exists()}
    if not bst:
        log.warning("%s: pixgbm — нет моделей в %s, пропущен", reg.name, md)
        return []
    out = []
    for mt in ("maggr", "etgr", "terfl"):
        files = pick_rasters(reg.rasters.get(mt, []))
        if not files:
            continue
        boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
        noise = geophys.load_mask(reg.noise, mt, crs_region)
        use = {k: v for k, v in bst.items() if k == "poly" or mt == "maggr"}
        for f in files:
            if stop is not None and stop():
                # участок целиком без pix: замена S идёт по (класс, метод) на весь участок, частичный
                # pix выбросил бы классику непройденных растров
                log.warning("%s: pixgbm — потолок времени перед %s, участок без pix (%d канд. отброшено)",
                            reg.name, f.name, len(out))
                return []
            try:
                d = load_map(f, GSD[mt], crs_region, boundary, noise)
                if not d["crs"]:
                    continue
                Pm = prob_map(d, mt, use)
                if "poly" in Pm:
                    out += poly_candidates(Pm["poly"], d, reg.name, mt, f.name)
                if "finds" in Pm:
                    out += finds_candidates(Pm["finds"], d, reg.name, mt, f.name)
                del d, Pm
            except Exception:
                log.exception("%s: pixgbm пропущен", f.name)
    return out


def dedupe_union(base: list[dict], new: list[dict], r_m: float = 1.0) -> list[dict]:
    """Правило U: из new убрать кандидатов, чей центр внутри полигона базы того же класса (object_poly)
    или ≤ r_m (EPSG:3857, как в оценке eval_pix.py) от центра находки базы (finds_points)."""
    from shapely.strtree import STRtree
    out = []
    by = {}
    for f in base:
        by.setdefault(f.get("class_name"), []).append(f["geom"])
    trees = {c: (STRtree(v), v) for c, v in by.items()}
    for f in new:
        t = trees.get(f["class_name"]); ce = f["geom"].centroid
        if t is not None:
            if f["class_name"] == "object_poly":
                hit = any(t[1][int(j)].contains(ce) for j in t[0].query(ce))
            else:
                hit = any(t[1][int(j)].centroid.distance(ce) <= r_m for j in t[0].query(ce.buffer(r_m)))
            if hit:
                continue
        out.append(f)
    return out
