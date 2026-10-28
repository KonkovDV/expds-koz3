"""Декодирование карт участка → таблица кандидатов (numpy/scipy, без torch).

Вход — вероятности (после sigmoid; у bot-9 predict_region уже даёт prob (5,H,W) после TTA-усреднения логитов):
  hm (2,h,w) f32 ∈ [0,1]; wh (1,h,w) f32 (код size_mode) или None; off (2,h,w) f32 или None — на сетке выхода
  головы (h = H/stride). Растр: верхний левый угол (left, top), шаг step = res·stride м/пикс (2,0 м при stride 4),
  СК метрическая (родная СК участка).
"""
from __future__ import annotations

import numpy as np

from .config import DEFAULTS
from .targets import size_decode

CAND_COLS = ("X", "Y", "cls", "score", "radius_m", "row", "col")


def find_peaks(p: np.ndarray, pool: int = 3, min_score: float = 0.05, topk: int = 20000,
               valid: np.ndarray | None = None):
    """p (H,W) → (rows, cols, scores), локальные максимумы окна pool×pool со score ≥ min_score, top-K по score.
    Плато (равные соседи) дают несколько пиков — их схлопывает nms_distance."""
    from scipy.ndimage import maximum_filter
    p = np.asarray(p, np.float32)
    mx = maximum_filter(p, size=pool, mode="constant", cval=-1.0)
    m = (p >= mx) & (p >= min_score)
    if valid is not None:
        m &= valid.astype(bool)
    r, c = np.nonzero(m)
    s = p[r, c]
    if len(s) > topk:
        i = np.argpartition(-s, topk - 1)[:topk]
        r, c, s = r[i], c[i], s[i]
    o = np.lexsort((c, r, -s))                  # детерминированный порядок: score ↓
    return r[o], c[o], s[o]


def nms_distance(X, Y, score, radius_m, cls, min_m: float = 1.0, frac: float = 0.8,
                 cross_class: bool = False) -> np.ndarray:
    """Жадное NMS по расстоянию в метрах: кандидат j подавляется более сильным i, если
    d(i,j) < max(min_m, frac·min(r_i, r_j)) и (cross_class или cls_i == cls_j). → индексы оставленных (score ↓)."""
    from scipy.spatial import cKDTree
    X, Y, score = map(np.asarray, (X, Y, score))
    n = len(X)
    if n == 0:
        return np.zeros(0, np.int64)
    radius_m = np.broadcast_to(np.asarray(radius_m, np.float64), (n,))
    cls = np.broadcast_to(np.asarray(cls), (n,))
    order = np.lexsort((np.arange(n), -score))
    xy = np.stack([X, Y], 1)
    tree = cKDTree(xy)
    rmax = max(min_m, frac * float(radius_m.max()))
    alive = np.ones(n, bool)
    keep = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(xy[i], rmax):
            if j == i or not alive[j]:
                continue
            if not cross_class and cls[j] != cls[i]:
                continue
            thr = max(min_m, frac * min(radius_m[i], radius_m[j]))
            if np.hypot(*(xy[j] - xy[i])) < thr:
                alive[j] = False
    return np.asarray(keep, np.int64)


def decode(hm: np.ndarray, left: float, top: float, step: float, wh: np.ndarray | None = None,
           off: np.ndarray | None = None, valid: np.ndarray | None = None, cfg: dict = DEFAULTS) -> dict:
    """→ таблица кандидатов dict[str, np.ndarray] с колонками CAND_COLS (X, Y — метры родной СК; radius_m — из
    головы размера или default_diam_m/2; row/col — целый центр-пиксель для выборки признаков)."""
    cols = {k: [] for k in CAND_COLS}
    for c in range(hm.shape[0]):
        r, cc, s = find_peaks(hm[c], cfg["peak_pool"], cfg["min_score"], cfg["topk"], valid)
        dx = off[0, r, cc] if off is not None else 0.0
        dy = off[1, r, cc] if off is not None else 0.0
        x = cc + 0.5 + dx
        y = r + 0.5 + dy
        rad = size_decode(wh[0, r, cc], cfg) * step if wh is not None else \
            np.full(len(r), cfg["default_diam_m"] / 2.0)
        cols["X"].append(left + x * step)
        cols["Y"].append(top - y * step)
        cols["cls"].append(np.full(len(r), c, np.int64))
        cols["score"].append(s.astype(np.float64))
        cols["radius_m"].append(np.asarray(rad, np.float64))
        cols["row"].append(r.astype(np.int64))
        cols["col"].append(cc.astype(np.int64))
    t = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}
    keep = nms_distance(t["X"], t["Y"], t["score"], t["radius_m"], t["cls"], cfg["nms_min_m"], cfg["nms_frac"],
                        cfg["nms_cross_class"])
    return take(t, keep)


def take(t: dict, idx) -> dict:
    return {k: np.asarray(v)[idx] for k, v in t.items()}


def concat(tables: list[dict]) -> dict:
    tables = [t for t in tables if t and len(next(iter(t.values())))]
    if not tables:
        return {k: np.zeros(0) for k in CAND_COLS}
    keys = tables[0].keys()
    return {k: np.concatenate([np.asarray(t[k]) for t in tables]) for k in keys}


def tile_origins(H: int, W: int, S: int = 512, step: int = 384, stride: int = DEFAULTS["stride"]):
    """Начала тайлов (y0, x0) для инференса со stride: как predict_region bot-9, но последний тайл выровнен вниз
    на кратное stride (иначе сетки выходов тайлов не совпадают). Требует step % stride == 0."""
    if step % stride or S % stride:
        raise ValueError("step и S должны делиться на stride")
    def ax(n):
        last = max(0, (n - S) // stride * stride)
        return sorted(set(list(range(0, max(1, n - S), step)) + [last]))
    return [(y, x) for y in ax(H) for x in ax(W)]


def stitch(outs, origins, H: int, W: int, S: int = 512, stride: int = DEFAULTS["stride"]) -> np.ndarray:
    """outs: список (K,S/s,S/s) (логиты/смещения после TTA), origins из tile_origins → среднее (K,ceil(H/s),ceil(W/s)).
    Хвост участка, не покрытый последним выровненным тайлом (< stride пикселей), остаётся 0 (cnt=0)."""
    h, w = -(-H // stride), -(-W // stride)
    K = outs[0].shape[0]
    acc = np.zeros((K, h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    n = S // stride
    for o, (y0, x0) in zip(outs, origins):
        ys, xs = y0 // stride, x0 // stride
        hh, ww = min(n, h - ys), min(n, w - xs)
        acc[:, ys:ys + hh, xs:xs + ww] += o[:, :hh, :ww]
        cnt[ys:ys + hh, xs:xs + ww] += 1
    return acc / np.maximum(cnt, 1)[None]
