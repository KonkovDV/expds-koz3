"""W4 Q6: кандидаты finds_points + вращательно-инвариантные дескрипторы для LightGBM (clean room).

Кандидаты для отдельного бустинга находок. В основной прогон не включены.
Кандидаты — локальные максимумы |hp| (hp = highpass(robust_z(карта), 2 м)) выше квантиля 0.88 с шагом ≥ 0.5 м.
Дескрипторы: кольцевые средние r 0.2…1.2 м, max/min/размах/СКО диска, произведения противоположных секторов (диполь),
асимметрия. По умолчанию нигде не включено.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import accel, geophys

RINGS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
RING_HW = 0.1
DISK_R = 1.2
SECT_IN, SECT_OUT, NSECT = 0.2, 0.8, 8
QUANT, MIN_DIST = 0.88, 0.5
TARGET_RES = 0.08

NAMES = (["c", "c_abs"] + [f"ring{r:.1f}" for r in RINGS] + [f"aring{r:.1f}" for r in RINGS]
         + ["dmax", "dmin", "drange", "dstd", "opp_min", "opp_mean", "dip_max", "asym", "qrank", "n2m", "logres"])


def decimate(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return a
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:h, :w].reshape(h // k, k, w // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3)).astype(np.float32)


def _offsets(res: float):
    rp = int(np.ceil(DISK_R / res))
    dy, dx = np.mgrid[-rp:rp + 1, -rp:rp + 1]
    d = np.hypot(dy, dx) * res
    th = np.arctan2(dy, dx)
    m = d <= DISK_R + 1e-9
    dy, dx, d, th = dy[m], dx[m], d[m], th[m]
    rings = [(np.abs(d - r) <= RING_HW + 1e-9) for r in RINGS]
    ann = (d >= SECT_IN) & (d <= SECT_OUT)
    sec = np.floor(((th + np.pi) / (2 * np.pi)) * NSECT).astype(int) % NSECT
    sects = [ann & (sec == k) for k in range(NSECT)]
    return dy, dx, rings, sects


def descriptors(hp: np.ndarray, rows: np.ndarray, cols: np.ndarray, res: float, chunk: int = 4000) -> np.ndarray:
    dy, dx, rings, sects = _offsets(res)
    H, W = hp.shape
    ab = np.abs(hp)
    fin = ab[np.isfinite(ab)]
    srt = np.sort(fin) if fin.size else np.zeros(1, np.float32)
    out = np.full((len(rows), len(NAMES)), np.nan, np.float32)
    ang = np.exp(1j * (np.arange(NSECT) + 0.5) * 2 * np.pi / NSECT)
    for s in range(0, len(rows), chunk):
        r, c = rows[s:s + chunk, None], cols[s:s + chunk, None]
        yy, xx = r + dy[None], c + dx[None]
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        v = np.where(ok, hp[np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)], np.nan).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"), _quiet():
            cen = hp[rows[s:s + chunk], cols[s:s + chunk]]
            f = [cen, np.abs(cen)]
            f += [np.nanmean(np.where(m[None], v, np.nan), axis=1) for m in rings]
            f += [np.nanmean(np.where(m[None], np.abs(v), np.nan), axis=1) for m in rings]
            mx, mn = np.nanmax(v, axis=1), np.nanmin(v, axis=1)
            f += [mx, mn, mx - mn, np.nanstd(v, axis=1)]
            sm = np.stack([np.nanmean(np.where(m[None], v, np.nan), axis=1) for m in sects], 1)
            opp = sm[:, :NSECT // 2] * sm[:, NSECT // 2:]
            f += [np.nanmin(opp, axis=1), np.nanmean(opp, axis=1),
                  np.nanmax(np.abs(sm[:, :NSECT // 2] - sm[:, NSECT // 2:]), axis=1)]
            smz = np.nan_to_num(sm)
            f += [np.abs(smz @ ang) / (np.abs(smz).sum(1) + 1e-6)]
            f += [np.searchsorted(srt, np.abs(cen)) / len(srt)]
            f += [np.zeros(len(cen)), np.full(len(cen), np.log(res))]
        out[s:s + chunk] = np.stack(f, 1)
    return out


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings(); self._w.__enter__(); warnings.simplefilter("ignore")

    def __exit__(self, *a):
        self._w.__exit__(*a)


def raster_candidates(path: str | Path, p: geophys.GeoParams, fallback_crs: str | None = None) -> dict:
    """Кандидаты одного растра: xy в CRS растра, дескрипторы, crs."""
    from affine import Affine
    from scipy.spatial import cKDTree

    from .raster import pixel_to_xy

    a, tr, crs, res = geophys.read_band(path, p.band_mode, fallback_crs)
    k = max(1, int(round(TARGET_RES / res)))
    if k > 1:
        a = decimate(a, k); tr = tr * Affine.scale(k); res = res * k
    hp = geophys.highpass(geophys.robust_z(a), res, p.background_m)
    del a
    ab = np.abs(hp)
    fin = ab[np.isfinite(ab)]
    if fin.size == 0:
        return dict(xy=np.zeros((0, 2)), X=np.zeros((0, len(NAMES)), np.float32), crs=crs, res=res)
    lo = float(np.quantile(fin, QUANT))
    rows, cols, _ = accel.find_peaks(ab, max(1, int(round(MIN_DIST / res))), lo)
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    X = descriptors(hp, rows, cols, res)
    xs, ys = pixel_to_xy(tr, rows, cols)
    xy = np.c_[xs, ys]
    if len(xy):
        X[:, NAMES.index("n2m")] = np.array([len(n) - 1 for n in cKDTree(xy).query_ball_point(xy, 2.0)], np.float32)
    return dict(xy=xy, X=X, crs=crs, res=res)


def nms(xy: np.ndarray, conf: np.ndarray, radius: float) -> np.ndarray:
    """Жадное подавление: индексы, оставшиеся в порядке убывания conf."""
    from scipy.spatial import cKDTree

    order = np.argsort(-conf, kind="stable")
    tree = cKDTree(xy)
    dead = np.zeros(len(xy), bool)
    keep = []
    for i in order:
        if dead[i]:
            continue
        keep.append(i)
        dead[tree.query_ball_point(xy[i], radius)] = True
    return np.asarray(keep, int)
