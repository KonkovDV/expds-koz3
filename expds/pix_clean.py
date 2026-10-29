"""Пиксельный LightGBM для карт геофизики — чистая перепись по docs/PIXGBM_SPEC_CLEANROOM.md.

Автор спецификации ≠ исполнитель (clean room): модуль написан только по спецификации,
документации numpy/scipy/lightgbm/rasterio и интерфейсу пайплайна. Включается
`[geophys.pixgbm] enabled = true, impl = "clean"`; по умолчанию не используется.

Схема:
1. Растр читается с уменьшением до рабочего шага (maggr 0,1 м, etgr 0,05 м, terfl — родной),
   фон за съёмкой (alpha, сплошной чёрный/белый, связанный с краем) — нет данных.
2. Цвет → скаляр (главная ось цвета), устойчивая нормализация (медиана/MAD), вычитание фона
   (нормированная свёртка Гаусса), признаки нескольких масштабов: Гаусс, DoG, LoG, градиент,
   Гессиан (det, λ1, λ2), локальное std, пиковость/впадинность, размах (диполь), аналитический
   сигнал, яркость, каналы, расстояние до края съёмки, метаданные.
3. Два бинарных LightGBM: finds (maggr) и poly (maggr/etgr/terfl, метод — признак).
4. Декодирование: finds — локальные максимумы вероятности с NMS радиуса R и субпиксельным
   центром; object_poly — связные компоненты p > t_pix, микрокруг в пике (или контур).
5. Срок фазы и лестница: полный → только finds (maggr) → классика; каждый откат — в лог,
   в COUNTERS и в status["fell"].
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

FEATURE_VERSION = 1
METHODS = ("maggr", "etgr", "terfl")
METHOD_ID = {"maggr": 0.0, "etgr": 1.0, "terfl": 2.0}
#: рабочий шаг сетки, м (None — родной шаг растра)
STEP_M = {"maggr": 0.1, "etgr": 0.05, "terfl": None}
#: фон для вычитания тренда, м
BACKGROUND_M = {"maggr": 2.0, "etgr": 2.0, "terfl": 6.0}
SIGMAS_PX = (1.0, 2.0, 4.0, 8.0)
TILE = 768
MARGIN = 64
MODEL_FILES = {"finds_points": "finds.txt", "object_poly": "poly.txt"}
CLASS_METHODS = {"finds_points": ("maggr",), "object_poly": ("maggr", "etgr", "terfl")}
DEFAULT_DECODE = {
    "nms_r_m": 1.0,          # радиус подавления соседних находок (§4.2: 0,5–1,5 м)
    "p_min": 0.02,           # минимальная вероятность пика находки
    "t_pix": 0.5,            # порог пикселя object_poly
    "min_area_m2": 0.1,      # мин. площадь компоненты object_poly
    "poly_mode": "peak",     # peak — микрокруг в пике компоненты; polygon — упрощённый контур
    "poly_peak_r_m": 0.25,
    "max_per_raster": 5000,
    "cross_map_r_m": 1.0,    # дедуп maggr между серой и цветной картой (один слой)
}
#: априорная скорость, с на Мпикс рабочей сетки (уточняется на ходу)
PRIOR_S_PER_MPIX = {"full": 6.0, "finds": 4.0}

#: счётчики откатов и событий (health)
COUNTERS: dict[str, int] = {}
_T0 = time.monotonic()


def _count(key: str, n: int = 1) -> None:
    COUNTERS[key] = COUNTERS.get(key, 0) + n


def strict() -> bool:
    return os.environ.get("EXPDS_PIXGBM_STRICT", "") == "1"


# ---------------------------------------------------------------- чтение растра

def feature_names() -> list[str]:
    names = ["z", "hp", "luma_hp"]
    for s in SIGMAS_PX:
        k = int(s)
        names += [f"g{k}", f"dog{k}", f"log{k}", f"grad{k}", f"hdet{k}", f"hl1_{k}", f"hl2_{k}",
                  f"std{k}", f"peak{k}", f"trough{k}", f"range{k}"]
    names += ["asig", "luma_g2", "luma_log2", "rgb_r", "rgb_g", "rgb_b", "edge_m",
              "method", "is_profile", "is_color", "res_native", "step", "n_bands"]
    return names


@dataclass
class Prep:
    """Растр на рабочей сетке: скаляр, яркость, каналы, маска данных и привязка."""
    path: Path
    mt: str
    hp: np.ndarray            # H×W float32, нормированный скаляр без фона (0 вне данных)
    z: np.ndarray             # H×W float32, нормированный скаляр
    luma: np.ndarray          # H×W float32, нормированная яркость без фона
    rgb: np.ndarray           # 3×H×W float32 в [0,1]
    valid: np.ndarray         # H×W bool
    edge_m: np.ndarray        # H×W float32
    transform: object         # affine рабочей сетки
    crs: str | None
    step: float               # шаг рабочей сетки в единицах CRS
    res_native: float
    n_bands: int
    is_color: bool
    is_profile: bool
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.hp.shape

    @property
    def mpix(self) -> float:
        return self.hp.size / 1e6


def _background(rgb_u: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Нет данных: прозрачное или сплошной чёрный/белый фон, связанный с краем растра."""
    lo = np.all(rgb_u <= 2.0, axis=0)
    hi = np.all(rgb_u >= 253.0, axis=0)
    bad = np.zeros(lo.shape, bool)
    for cand in (lo, hi):
        if not cand.any():
            continue
        lab, n = ndi.label(cand, structure=np.ones((3, 3), bool))
        if n == 0:
            continue
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        edge = edge[edge > 0]
        if edge.size:
            bad |= np.isin(lab, edge)
    if alpha is not None:
        bad |= alpha <= 0.5
    return bad


def _robust(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = a[valid]
    if v.size < 16:
        return np.zeros_like(a, dtype=np.float32)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if not mad > 1e-6:
        sd = float(np.std(v))
        mad = sd if sd > 1e-6 else 1.0
    out = (a - med) / mad
    return np.where(valid, np.clip(out, -20, 20), 0.0).astype(np.float32)


def _norm_gauss(a: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Гаусс только по пикселям с данными (нормированная свёртка)."""
    w = valid.astype(np.float32)
    num = ndi.gaussian_filter(np.where(valid, a, 0).astype(np.float32), sigma, mode="nearest")
    den = ndi.gaussian_filter(w, sigma, mode="nearest")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0).astype(np.float32)


def _pca_axis(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = rgb[:, valid].astype(np.float64)
    if x.shape[1] < 16 or np.allclose(x.std(axis=1), 0):
        return np.array([1.0, 0.0, 0.0])
    step = max(1, x.shape[1] // 1_000_000)
    s = x[:, ::step]
    cov = np.cov(s - s.mean(axis=1, keepdims=True))
    vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
    axis = vecs[:, int(np.argmax(vals))]
    if float(axis @ np.array([0.299, 0.587, 0.114])) < 0:
        axis = -axis
    return axis


def is_profile(path: Path) -> bool:
    return "профил" in path.stem.lower()


def is_color(path: Path) -> bool:
    return "цвет" in path.stem.lower()


def prepare(path: str | Path, mt: str, fallback_crs: str | None = None,
            step_m: float | None = None) -> Prep:
    """Читает растр на рабочей сетке и считает общие для всех признаков слои."""
    import rasterio
    from rasterio.enums import Resampling

    from .crs import usable_raster_crs
    from .raster import open_metric

    path = Path(path)
    if step_m is None:
        step_m = STEP_M.get(mt)
    with open_metric(path) as ds:
        res = float(abs(ds.res[0]))
        f = 1 if not step_m else max(1, int(round(step_m / res)))
        h, w = max(1, ds.height // f), max(1, ds.width // f)
        n = ds.count
        alpha_idx = [i + 1 for i, ci in enumerate(ds.colorinterp) if getattr(ci, "name", "") == "alpha"]
        color_idx = [i for i in range(1, n + 1) if i not in alpha_idx][:3] or [1]
        data = ds.read(color_idx, out_shape=(len(color_idx), h, w),
                       resampling=Resampling.average if f > 1 else Resampling.nearest).astype(np.float32)
        alpha = None
        if alpha_idx:
            alpha = ds.read(alpha_idx[0], out_shape=(h, w), resampling=Resampling.nearest).astype(np.float32)
            alpha = alpha / max(1.0, float(np.nanmax(alpha)) if alpha.size else 1.0)
        msk = None
        if ds.nodata is not None:
            msk = ds.read_masks(color_idx[0], out_shape=(h, w), resampling=Resampling.nearest) == 0
        transform = ds.transform @ rasterio.Affine.scale(ds.width / w, ds.height / h)
        crs = usable_raster_crs(ds.crs.to_string() if ds.crs else None, ds.bounds, fallback_crs)
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] == 2:
        data = np.concatenate([data, data[1:2]], axis=0)
    bad = _background(data, alpha)
    if msk is not None:
        bad |= msk
    valid = ~bad & np.all(np.isfinite(data), axis=0)
    del bad
    np.divide(data, 255.0, out=data)
    data[:, ~valid] = 0.0
    rgb = data                      # без копии: на 14 Мпикс это 170 МБ
    del data
    axis = _pca_axis(rgb, valid)
    scal = np.tensordot(axis.astype(np.float32), rgb, axes=1)
    luma = np.tensordot(np.array([0.299, 0.587, 0.114], np.float32), rgb, axes=1)
    step = float(abs(transform.a))
    bg = max(2.0, BACKGROUND_M.get(mt, 2.0) / step)
    z = _robust(scal, valid)
    del scal
    hp = _robust(z - _norm_gauss(z, valid, bg), valid)
    lz = _robust(luma, valid)
    del luma
    lhp = _robust(lz - _norm_gauss(lz, valid, bg), valid)
    edge = (ndi.distance_transform_edt(valid) * step).astype(np.float32) if valid.any() else np.zeros(valid.shape, np.float32)
    edge = np.minimum(edge, 30.0)
    return Prep(path=path, mt=mt, hp=hp, z=z, luma=lhp, rgb=rgb, valid=valid, edge_m=edge,
                transform=transform, crs=crs, step=step, res_native=res, n_bands=int(n),
                is_color=is_color(path), is_profile=is_profile(path))


# ---------------------------------------------------------------- признаки

def _tile_features(p: Prep, r0: int, r1: int, c0: int, c1: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Признаки окна [r0:r1, c0:c1] (с полями MARGIN для фильтров). Возвращает F×h×w и срез ядра."""
    H, W = p.shape
    R0, R1 = max(0, r0 - MARGIN), min(H, r1 + MARGIN)
    C0, C1 = max(0, c0 - MARGIN), min(W, c1 + MARGIN)
    hp = p.hp[R0:R1, C0:C1]
    feats = [p.z[R0:R1, C0:C1], hp, p.luma[R0:R1, C0:C1]]
    for s in SIGMAS_PX:
        g = ndi.gaussian_filter(hp, s, mode="nearest")
        g2 = ndi.gaussian_filter(hp, 2 * s, mode="nearest")
        hxx = ndi.gaussian_filter(hp, s, order=(0, 2), mode="nearest")
        hyy = ndi.gaussian_filter(hp, s, order=(2, 0), mode="nearest")
        hxy = ndi.gaussian_filter(hp, s, order=(1, 1), mode="nearest")
        gx = ndi.gaussian_filter(hp, s, order=(0, 1), mode="nearest")
        gy = ndi.gaussian_filter(hp, s, order=(1, 0), mode="nearest")
        s2 = s * s
        tr = (hxx + hyy) * 0.5
        disc = np.sqrt(((hxx - hyy) * 0.5) ** 2 + hxy ** 2)
        win = int(2 * s + 1)
        m1 = ndi.uniform_filter(hp, win, mode="nearest")
        m2 = ndi.uniform_filter(hp * hp, win, mode="nearest")
        big = int(6 * s + 1)
        mx = ndi.maximum_filter(g, big, mode="nearest")
        mn = ndi.minimum_filter(g, big, mode="nearest")
        feats += [g, g - g2, s2 * (hxx + hyy), s * np.hypot(gx, gy), s2 * s2 * (hxx * hyy - hxy * hxy),
                  s2 * (tr + disc), s2 * (tr - disc), np.sqrt(np.maximum(m2 - m1 * m1, 0)),
                  mx - g, g - mn, mx - mn]
    feats.append(_analytic(hp, p.valid[R0:R1, C0:C1]))
    lu = p.luma[R0:R1, C0:C1]
    feats += [ndi.gaussian_filter(lu, 2, mode="nearest"), 4 * ndi.gaussian_laplace(lu, 2, mode="nearest")]
    for c in range(3):
        feats.append(ndi.gaussian_filter(p.rgb[c, R0:R1, C0:C1], 1, mode="nearest"))
    feats.append(p.edge_m[R0:R1, C0:C1])
    shp = hp.shape
    for v in (METHOD_ID.get(p.mt, 0.0), float(p.is_profile), float(p.is_color), p.res_native, p.step,
              float(p.n_bands)):
        feats.append(np.full(shp, v, np.float32))
    arr = np.stack([np.asarray(a, np.float32) for a in feats])
    return arr, (r0 - R0, r1 - R0, c0 - C0, c1 - C0)


def _analytic(hp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Амплитуда аналитического сигнала (производные в спектре, лёгкое продолжение вверх на 1 пиксель)."""
    f = np.where(valid, hp, 0.0).astype(np.float32)
    ny, nx = f.shape
    py, px = ny + ny // 8 + 8, nx + nx // 8 + 8
    spec = np.fft.rfft2(f, s=(py, px))
    ky = 2 * np.pi * np.fft.fftfreq(py)[:, None]
    kx = 2 * np.pi * np.fft.rfftfreq(px)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    spec = spec * np.exp(-k)
    gx = np.fft.irfft2(spec * (1j * kx), s=(py, px))[:ny, :nx]
    gy = np.fft.irfft2(spec * (1j * ky), s=(py, px))[:ny, :nx]
    gz = np.fft.irfft2(spec * k, s=(py, px))[:ny, :nx]
    return np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)


def iter_tiles(shape: tuple[int, int], tile: int = TILE):
    H, W = shape
    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            yield r0, min(H, r0 + tile), c0, min(W, c0 + tile)


def features_at(p: Prep, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Признаки выбранных пикселей (N×F), считаются по окнам, где эти пиксели есть."""
    nf = len(feature_names())
    out = np.zeros((len(rows), nf), np.float32)
    if not len(rows):
        return out
    tr, tc = rows // TILE, cols // TILE
    key = tr * (p.shape[1] // TILE + 1) + tc
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    for grp in np.split(order, bounds):
        r0, c0 = int(tr[grp[0]]) * TILE, int(tc[grp[0]]) * TILE
        r1, c1 = min(p.shape[0], r0 + TILE), min(p.shape[1], c0 + TILE)
        arr, (a0, _, b0, _) = _tile_features(p, r0, r1, c0, c1)
        out[grp] = arr[:, rows[grp] - r0 + a0, cols[grp] - c0 + b0].T
    return out


def predict_maps(p: Prep, models: dict, threads: int = 1, deadline: float | None = None) -> dict[str, np.ndarray]:
    """Карты вероятности по классам (H×W float32, 0 вне данных). deadline — time.monotonic()."""
    out = {c: np.zeros(p.shape, np.float32) for c in models}
    for r0, r1, c0, c1 in iter_tiles(p.shape):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("срок фазы pix_clean истёк внутри растра")
        v = p.valid[r0:r1, c0:c1]
        if not v.any():
            continue
        arr, (a0, a1, b0, b1) = _tile_features(p, r0, r1, c0, c1)
        core = arr[:, a0:a1, b0:b1]
        X = core[:, v].T
        for c, booster in models.items():
            pr = booster.predict(X, num_threads=max(1, int(threads)))
            tile = np.zeros(v.shape, np.float32)
            tile[v] = pr
            out[c][r0:r1, c0:c1] = tile
    return out


# ---------------------------------------------------------------- декодирование

def _xy(p: Prep, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from .raster import pixel_to_xy

    return pixel_to_xy(p.transform, np.asarray(rows, np.float64), np.asarray(cols, np.float64))


def decode_finds(prob: np.ndarray, p: Prep, nms_r_m: float, p_min: float, max_n: int = 5000) -> list[tuple]:
    """Локальные максимумы вероятности: [(x, y, conf)] в CRS растра, центр — взвешенный 3×3 (субпиксель)."""
    sm = ndi.gaussian_filter(prob, 1.0, mode="nearest")
    rad = max(1, int(round(nms_r_m / p.step)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    foot = (yy * yy + xx * xx) <= rad * rad
    mx = ndi.maximum_filter(sm, footprint=foot, mode="constant", cval=0.0)
    peaks = (sm >= mx) & (sm >= p_min) & p.valid
    r, c = np.nonzero(peaks)
    if not len(r):
        return []
    conf = prob[r, c].astype(np.float64)
    order = np.lexsort((c, r, -sm[r, c]))[: max_n * 4]
    r, c, conf = r[order], c[order], conf[order]
    # плато: равные соседние максимумы — один пик (жадно по порядку)
    keep, taken = [], np.zeros(len(r), bool)
    pts = np.stack([r, c], 1).astype(np.float64)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    for i in range(len(r)):
        if taken[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(pts[i], rad):
            taken[j] = True
        if len(keep) >= max_n:
            break
    out = []
    H, W = prob.shape
    for i in keep:
        r0, c0 = int(r[i]), int(c[i])
        ra, rb, ca, cb = max(0, r0 - 1), min(H, r0 + 2), max(0, c0 - 1), min(W, c0 + 2)
        w = sm[ra:rb, ca:cb].astype(np.float64)
        w = np.maximum(w - w.min(), 0) + 1e-9
        gy, gx = np.mgrid[ra:rb, ca:cb]
        rr, cc = float((w * gy).sum() / w.sum()), float((w * gx).sum() / w.sum())
        x, y = _xy(p, np.array([rr]), np.array([cc]))
        out.append((float(x[0]), float(y[0]), float(np.clip(sm[r0, c0], 0, 1))))
    return out


def decode_poly(prob: np.ndarray, p: Prep, t_pix: float, min_area_m2: float, mode: str = "peak",
                max_n: int = 5000) -> list[tuple]:
    """Компоненты p > t_pix: [(x, y, conf, labels_or_None, label_id)] в CRS растра."""
    sm = ndi.gaussian_filter(prob, 1.0, mode="nearest")
    m = (sm > t_pix) & p.valid
    lab, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(sm), lab, idx) * p.step * p.step
    peak = ndi.maximum_position(sm, lab, idx)
    pmax = ndi.maximum(sm, lab, idx)
    out = []
    for i, k in enumerate(idx):
        if area[i] < min_area_m2:
            continue
        r0, c0 = peak[i]
        x, y = _xy(p, np.array([r0]), np.array([c0]))
        out.append((float(x[0]), float(y[0]), float(np.clip(pmax[i], 0, 1)), int(k)))
    out.sort(key=lambda t: -t[2])
    out = out[:max_n]
    if mode == "polygon" and out:
        keep_ids = {t[3] for t in out}
        lab = np.where(np.isin(lab, list(keep_ids)), lab, 0)
        return [(x, y, cf, k, lab) for x, y, cf, k in out]
    return [(x, y, cf, k, None) for x, y, cf, k in out]


# ---------------------------------------------------------------- модели

def load_models(model_dir: str | Path, classes) -> tuple[dict, dict]:
    """{класс: Booster}, meta. Нет файла: строгий режим — исключение, иначе класс пропускается."""
    model_dir = Path(model_dir)
    models, meta = {}, {}
    mp = model_dir / "meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    try:
        import lightgbm as lgb
    except ImportError:
        if strict():
            raise
        _count("no_lightgbm")
        log.warning("pix_clean: lightgbm не установлен — классика")
        return {}, meta
    for c in classes:
        f = model_dir / MODEL_FILES[c]
        if not f.exists():
            if strict():
                raise FileNotFoundError(f"pix_clean: нет модели {f} (EXPDS_PIXGBM_STRICT=1)")
            _count("no_model")
            log.warning("pix_clean: нет модели %s — класс %s остаётся на классике", f, c)
            continue
        models[c] = lgb.Booster(model_file=str(f))
        nf = models[c].num_feature()
        if nf != len(feature_names()):
            msg = f"pix_clean: {f.name}: {nf} признаков, ожидается {len(feature_names())}"
            if strict():
                raise ValueError(msg)
            _count("bad_model")
            log.warning("%s — класс %s остаётся на классике", msg, c)
            models.pop(c)
    return models, meta


_MODEL_CACHE: dict = {}


def _models_cached(model_dir: Path, classes: tuple) -> tuple[dict, dict]:
    key = (str(Path(model_dir).resolve()), classes)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_models(model_dir, classes)
    return _MODEL_CACHE[key]


# ---------------------------------------------------------------- срок фазы

def _budget_s(pix: dict) -> float:
    env = os.environ.get("PIXGBM_BUDGET_S")
    if env:
        try:
            return float(env)
        except ValueError:
            log.warning("PIXGBM_BUDGET_S=%r — не число, беру из конфига", env)
    return float(pix.get("budget_s", 600))


def time_left(pix: dict, guard=None, now: float | None = None) -> float:
    """Секунд до конца фазы: T0 + budget_s и guard.left() − guard_reserve_s, что раньше."""
    if guard is not None:
        left = _budget_s(pix) - guard.elapsed()
        left = min(left, guard.left() - float(pix.get("guard_reserve_s", 0)))
        return left
    now = time.monotonic() if now is None else now
    return _budget_s(pix) - (now - _T0)


_RATE: dict[str, float] = {}


def forecast_s(mode: str, mpix: float) -> float:
    return 1.2 * _RATE.get(mode, PRIOR_S_PER_MPIX[mode]) * mpix + 0.5


def _learn_rate(mode: str, mpix: float, secs: float) -> None:
    if mpix <= 0.01:
        return
    r = secs / mpix
    old = _RATE.get(mode)
    _RATE[mode] = r if old is None else 0.5 * old + 0.5 * r


def estimate_mpix(path: Path, mt: str) -> float:
    import rasterio

    with rasterio.open(path) as ds:
        res = float(abs(ds.res[0])) or 1.0
        step = STEP_M.get(mt)
        f = 1 if not step else max(1, int(round(step / res)))
        return (ds.width // f) * (ds.height // f) / 1e6


# ---------------------------------------------------------------- участок

def _threads(pix: dict) -> int:
    n = int(pix.get("procs", 0) or 0)
    if n <= 0:
        try:
            n = min(len(os.sched_getaffinity(0)), 4)
        except AttributeError:
            n = min(os.cpu_count() or 1, 4)
    return max(1, n)


def _feat(geom, cls, mt, reg_name, src, crs, conf, x=None, y=None) -> dict:
    f = {"geom": geom, "class_name": cls, "confidence": float(np.clip(conf, 0.0, 1.0)),
         "region_name": reg_name, "markup_type": mt, "source": src, "original_crs": crs,
         "threshold_group": f"{mt}_pix"}
    if x is not None:
        f["_xy"] = (float(x), float(y), crs)
    return f


def raster_feats(p: Prep, probs: dict, dec: dict, reg_name: str, point_r_m: float) -> list[dict]:
    """Кандидаты одного растра по картам вероятности."""
    from .raster import circle_3857, mask_to_polygons

    out = []
    if "finds_points" in probs:
        for x, y, cf in decode_finds(probs["finds_points"], p, float(dec["nms_r_m"]), float(dec["p_min"]),
                                     int(dec["max_per_raster"])):
            out.append(_feat(circle_3857(x, y, point_r_m, p.crs), "finds_points", p.mt, reg_name,
                             p.path.name, p.crs, cf, x, y))
    if "object_poly" in probs:
        comps = decode_poly(probs["object_poly"], p, float(dec["t_pix"]), float(dec["min_area_m2"]),
                            str(dec["poly_mode"]), int(dec["max_per_raster"]))
        polys = {}
        if comps and comps[0][4] is not None:
            polys = {k: g for g, k in mask_to_polygons(comps[0][4], p.transform, p.crs, simplify=p.step * 0.5)}
        for x, y, cf, k, _ in comps:
            g = polys.get(k) if polys else None
            if g is None:
                g = circle_3857(x, y, float(dec["poly_peak_r_m"]), p.crs)
            out.append(_feat(g, "object_poly", p.mt, reg_name, p.path.name, p.crs, cf, x, y))
    return out


def _ground_xy(f: dict) -> tuple[float, float]:
    """Центр объекта в метрах на местности (локально): EPSG:3857 × cos(широта)."""
    c = f["geom"].centroid
    lat = math.atan(math.sinh(c.y / 6_378_137.0))
    k = math.cos(lat)
    return c.x * k, c.y * k


def dedupe_cross_maps(feats: list[dict], r_m: float) -> list[dict]:
    """Серая и цветная карта MagGr — один слой: из близких (≤ r_m) объектов разных карт остаётся лучший."""
    from scipy.spatial import cKDTree

    out = []
    groups: dict[tuple, list[dict]] = {}
    for f in feats:
        groups.setdefault((f["class_name"], f["markup_type"]), []).append(f)
    for (cls, mt), items in groups.items():
        if mt == "etgr" or len({f["source"] for f in items}) < 2:
            out += items
            continue
        items = sorted(items, key=lambda f: -f["confidence"])
        pts = np.array([_ground_xy(f) for f in items])
        tree = cKDTree(pts)
        dead = np.zeros(len(items), bool)
        for i, f in enumerate(items):
            if dead[i]:
                continue
            out.append(f)
            for j in tree.query_ball_point(pts[i], r_m):
                if j != i and items[j]["source"] != f["source"]:
                    dead[j] = True
    return out


def select_files(reg, mt: str, pix: dict) -> list[Path]:
    """Растры метода. maggr: все карты (серая и цветная — один слой, дедуп после); etgr: каждый слой."""
    files = list(reg.rasters.get(mt, []) or [])
    if mt == "etgr" and pix.get("etgr_max_depth_m") is not None:
        lim = float(pix["etgr_max_depth_m"])
        keep = []
        for f in files:
            m = re.search(r"карта_\d+_(\d+(?:[.,]\d+)?)", f.stem)
            if m is None or float(m.group(1).replace(",", ".")) <= lim:
                keep.append(f)
        files = keep
    return sorted(files, key=lambda f: f.name)


def region_feats(reg, cfg: dict, model_dir, guard=None, status: dict | None = None) -> list[dict]:
    """Кандидаты pix_clean участка. status заполняется: done[(класс, метод)] — растры, fell[(класс, метод)] — откаты."""
    from . import geophys

    if status is None:
        status = {}
    done = status.setdefault("done", {})
    fell = status.setdefault("fell", {})
    pix = ((cfg or {}).get("geophys") or {}).get("pixgbm") or {}
    geo_cfg = (cfg or {}).get("geophys") or {}
    classes = tuple(sorted(set(pix.get("classes", ["finds_points", "object_poly"]))))
    maps = pix.get("maps", list(METHODS))
    if isinstance(maps, str):
        maps = [maps]
    models, meta = _models_cached(Path(model_dir), classes)
    dec = dict(DEFAULT_DECODE, **(meta.get("decode") or {}))
    for k in DEFAULT_DECODE:
        if k in pix:
            dec[k] = pix[k]
    threads = _threads(pix)
    point_r = float(geo_cfg.get("point_circle_m", 0.3))
    try:
        from .pipeline import region_crs

        crs_region = region_crs(reg, cfg)
    except Exception:
        crs_region = None
    out: list[dict] = []
    for mt in METHODS:
        if mt not in maps:
            continue
        mt_classes = [c for c in classes if mt in CLASS_METHODS[c]]
        if not mt_classes:
            continue
        files = select_files(reg, mt, pix)
        if not files:
            continue
        have = [c for c in mt_classes if c in models]
        for c in mt_classes:
            if c not in have:
                fell[(c, mt)] = True
        if not have:
            continue
        try:
            boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
            noise = geophys.load_mask(reg.noise, mt, crs_region)
        except Exception:
            log.exception("pix_clean: %s %s: маски не прочитаны — без масок", reg.name, mt)
            _count("mask_error")
            boundary = noise = None
        mine: list[dict] = []
        for path in files:
            try:
                mpix = estimate_mpix(path, mt)
            except Exception:
                mpix = 1.0
            left = time_left(pix, guard)
            run = list(have)
            mode = "full"
            if forecast_s("full", mpix) > left:
                if mt == "maggr" and "finds_points" in have and forecast_s("finds", mpix) <= left:
                    run, mode = ["finds_points"], "finds"
                    log.warning("pix_clean: %s: срок фазы — только finds (прогноз %.0f с, осталось %.0f с)",
                                path.name, forecast_s("full", mpix), left)
                    _count("ladder_finds_only")
                else:
                    run = []
                    log.warning("pix_clean: %s: срок фазы — классика (прогноз %.0f с, осталось %.0f с)",
                                path.name, forecast_s(mode, mpix), left)
                    _count("ladder_classic")
            for c in have:
                if c not in run:
                    fell[(c, mt)] = True
            if not run:
                continue
            t = time.monotonic()
            try:
                p = prepare(path, mt, crs_region)
                probs = predict_maps(p, {c: models[c] for c in run}, threads,
                                     deadline=time.monotonic() + max(1.0, time_left(pix, guard)))
                feats = raster_feats(p, probs, dec, reg.name, point_r)
                b_map, n_map = geophys.sane_masks(path, boundary, noise, geo_cfg, crs_region)
                feats = geophys.apply_masks(feats, b_map, n_map)
            except Exception as exc:
                log.warning("pix_clean: %s: ошибка (%s: %s) — классика на этом растре", path.name,
                            type(exc).__name__, exc)
                _count("raster_error")
                for c in run:
                    fell[(c, mt)] = True
                continue
            _learn_rate(mode, p.mpix, time.monotonic() - t)
            for c in run:
                done.setdefault((c, mt), set()).add(path.name)
            mine += feats
        if mt == "maggr":
            mine = dedupe_cross_maps(mine, float(dec["cross_map_r_m"]))
        out += mine
    for f in out:
        f.pop("_xy", None)
    return out


# ---------------------------------------------------------------- объединение (вариант H/U)

def dedupe_union(old_feats: list[dict], new_feats: list[dict]) -> list[dict]:
    """new без дублей old: тот же класс и метод (EtGr — тот же слой). object_poly — центр нового внутри
    старого полигона; finds_points — ≤ 1 м на местности."""
    from shapely.prepared import prep
    from shapely.strtree import STRtree

    def key(f):
        mt = f.get("markup_type")
        return (f.get("class_name"), mt, f.get("source") if mt == "etgr" else None)

    by: dict[tuple, list[dict]] = {}
    for f in old_feats:
        if f.get("class_name") in ("object_poly", "finds_points"):
            by.setdefault(key(f), []).append(f)
    trees: dict = {}
    out = []
    for f in new_feats:
        olds = by.get(key(f))
        if not olds:
            out.append(f)
            continue
        k = key(f)
        if f["class_name"] == "object_poly":
            if k not in trees:
                geoms = [o["geom"] for o in olds]
                trees[k] = (STRtree(geoms), [prep(g) for g in geoms])
            tree, preps = trees[k]
            c = f["geom"].representative_point() if f["geom"].geom_type != "Point" else f["geom"]
            dup = any(preps[int(i)].contains(c) for i in tree.query(c))
        else:
            if k not in trees:
                trees[k] = np.array([_ground_xy(o) for o in olds])
            pts = trees[k]
            x, y = _ground_xy(f)
            dup = bool(np.any((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2 <= 1.0))
        if not dup:
            out.append(f)
    return out
