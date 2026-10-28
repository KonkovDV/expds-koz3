"""Признаки кандидатов для реранкера. Все признаки — из карт/рельефа самого участка (никаких меток GT).

Патч-выборка вокруг центр-пикселя (row, col), полуокно w = cfg["feat_win_px"] (21×21 при w=10), чанками
(память ~ chunk·C·(2w+1)² · 4 Б). Квадратные «диски» (Чебышёв), кольцо = окно w минус окно r_in.
"""
from __future__ import annotations

import numpy as np

from .config import DEFAULTS, LI_TAGS12


def _patches(a: np.ndarray, rows, cols, w: int, fill=np.nan) -> np.ndarray:
    """a (C,H,W) → (N,C,2w+1,2w+1) float32, за краем — fill."""
    C, H, W = a.shape
    n = len(rows)
    out = np.full((n, C, 2 * w + 1, 2 * w + 1), fill, np.float32)
    d = np.arange(-w, w + 1)
    rr = rows[:, None] + d[None]
    cc = cols[:, None] + d[None]
    okr = (rr >= 0) & (rr < H)
    okc = (cc >= 0) & (cc < W)
    rr_c = np.clip(rr, 0, H - 1)
    cc_c = np.clip(cc, 0, W - 1)
    for c in range(C):
        v = np.asarray(a[c])[rr_c[:, :, None], cc_c[:, None, :]].astype(np.float32)
        v[~(okr[:, :, None] & okc[:, None, :])] = fill
        out[:, c] = v
    return out


def _win_mean(p, r):
    w = p.shape[-1] // 2
    return np.nanmean(p[..., w - r:w + r + 1, w - r:w + r + 1], axis=(-1, -2))


def _ring_mean(p, r_in):
    w = p.shape[-1] // 2
    q = p.copy()
    q[..., w - r_in:w + r_in + 1, w - r_in:w + r_in + 1] = np.nan
    return np.nanmean(q, axis=(-1, -2))


def neighbour_features(t: dict, radii_m=DEFAULTS["nb_radii_m"]) -> tuple[np.ndarray, list[str]]:
    """Априор соседей по всем кандидатам участка (любой класс): число, сумма и максимум score в радиусе R;
    расстояние/score ближайшего, расстояние до ближайшего кандидата другого класса."""
    from scipy.spatial import cKDTree
    X, Y, s, c = (np.asarray(t[k]) for k in ("X", "Y", "score", "cls"))
    n = len(X)
    names, cols = [], []
    if n == 0:
        for R in radii_m:
            names += [f"nb_cnt_{R:g}", f"nb_sum_{R:g}", f"nb_max_{R:g}"]
        names += ["nn_dist", "nn_score", "nn_other_cls_dist"]
        return np.zeros((0, len(names)), np.float32), names
    xy = np.stack([X, Y], 1)
    tree = cKDTree(xy)
    for R in radii_m:
        lst = tree.query_ball_point(xy, R)
        cnt = np.array([len(l) - 1 for l in lst], np.float32)
        sm = np.array([s[l].sum() - s[i] for i, l in enumerate(lst)], np.float32)
        mx = np.array([max([s[j] for j in l if j != i], default=0.0) for i, l in enumerate(lst)], np.float32)
        cols += [cnt, sm, mx]
        names += [f"nb_cnt_{R:g}", f"nb_sum_{R:g}", f"nb_max_{R:g}"]
    big = 1e4
    if n > 1:
        d, j = tree.query(xy, k=2)
        cols += [d[:, 1].astype(np.float32), s[j[:, 1]].astype(np.float32)]
    else:
        cols += [np.full(n, big, np.float32), np.zeros(n, np.float32)]
    oth = np.full(n, big, np.float32)
    for k in np.unique(c):
        a, b = c == k, c != k
        if b.any():
            oth[a] = cKDTree(xy[b]).query(xy[a])[0]
    cols.append(oth)
    names += ["nn_dist", "nn_score", "nn_other_cls_dist"]
    return np.stack(cols, 1), names


def gor_poly_features(t: dict, gor_polys) -> tuple[np.ndarray, list[str]]:
    """Расстояние (м) до предсказанных полигонов городищ в родной СК: >0 снаружи, <0 внутри (до границы)."""
    from shapely import STRtree, points, distance, contains_xy
    X, Y = np.asarray(t["X"]), np.asarray(t["Y"])
    n = len(X)
    if not gor_polys or n == 0:
        return np.column_stack([np.full(n, 1e4, np.float32), np.zeros(n, np.float32)]), ["gor_sdist_m", "in_gor"]
    polys = list(gor_polys)
    tree = STRtree(polys)
    pts = points(np.stack([X, Y], 1))
    idx = tree.nearest(pts)
    near = np.array(polys, dtype=object)[idx]
    inside = np.array([contains_xy(g, x, y) for g, x, y in zip(near, X, Y)], bool)
    # точка может лежать внутри другого (не ближайшего по границе) — ближайший по distance=0 и есть содержащий
    d = np.array([distance(g.boundary, p) for g, p in zip(near, pts)], np.float32)
    sd = np.where(inside, -d, np.array([distance(g, p) for g, p in zip(near, pts)], np.float32))
    return np.column_stack([sd.astype(np.float32), inside.astype(np.float32)]), ["gor_sdist_m", "in_gor"]


def rc_on_grid(t: dict, grid) -> tuple[np.ndarray, np.ndarray]:
    """grid = (left, top, step) растра → целые (row, col) центров кандидатов на этом растре."""
    left, top, step = grid
    return (np.floor((top - np.asarray(t["Y"], np.float64)) / step).astype(np.int64),
            np.floor((np.asarray(t["X"], np.float64) - left) / step).astype(np.int64))


def build_features(t: dict, hm: np.ndarray, mask: np.ndarray | None = None, relief: np.ndarray | None = None,
                   gor_polys=None, hm_grid=None, full_grid=None,
                   relief_names=LI_TAGS12, cfg: dict = DEFAULTS, chunk: int = 2000) -> tuple[np.ndarray, list[str]]:
    """t — таблица decode(); hm (2,h,w) prob на сетке головы; mask (3,H,W) prob [gor, fort, arch] и relief (C,H,W)
    uint8 (0 = nodata, C = 3/5/12 — имена берутся как relief_names[:C]) — на полной сетке (0,5 м).
    hm_grid / full_grid = (left, top, step): если заданы, row/col считаются из X, Y; иначе берутся t["row"], t["col"]
    (тогда все карты на одной сетке). → (F float32 (N,n_feat), names). NaN допустимы (LightGBM их понимает).
    Окна признаков (feat_win_px и производные) — в пикселях своей сетки."""
    rows, cols = rc_on_grid(t, hm_grid) if hm_grid is not None else \
        (np.asarray(t["row"], np.int64), np.asarray(t["col"], np.int64))
    frows, fcols = rc_on_grid(t, full_grid) if full_grid is not None else (rows, cols)
    n = len(rows)
    w = int(cfg["feat_win_px"])
    blocks, names = [], []
    cls = np.asarray(t["cls"])
    base = np.column_stack([np.asarray(t["score"], np.float32), cls.astype(np.float32),
                            np.asarray(t["radius_m"], np.float32)])
    blocks.append(base)
    names += ["score", "cls", "radius_m"]
    import warnings
    hm_f, rel_f, m_f = [], [], []
    _w = warnings.catch_warnings()
    _w.__enter__()
    warnings.simplefilter("ignore", RuntimeWarning)        # nanmean пустых окон за краем → NaN (ожидаемо)
    for a in range(0, n, chunk):
        r, c = rows[a:a + chunk], cols[a:a + chunk]
        fr, fc = frows[a:a + chunk], fcols[a:a + chunk]
        ph = _patches(hm, r, c, w)                                  # (n,2,·,·)
        k = cls[a:a + chunk]
        own = ph[np.arange(len(r)), k]
        oth = ph[np.arange(len(r)), 1 - k] if hm.shape[0] == 2 else own
        ctr = own[:, w, w]
        hm_f.append(np.column_stack([
            oth[:, w, w], _win_mean(own, 1), _win_mean(own, 3), ctr - _ring_mean(own, 3),
            (own >= 0.5 * ctr[:, None, None]).sum((1, 2)).astype(np.float32),   # «ширина» пика: пикс ≥ ½ высоты
            np.nanmax(oth, (1, 2)),
        ]))
        if mask is not None:
            pm = _patches(mask, fr, fc, w)
            m_f.append(np.column_stack([pm[:, 0, w, w], np.nanmax(pm[:, 0], (1, 2)), pm[:, 1, w, w],
                                        np.nanmax(pm[:, 1], (1, 2)), pm[:, 2, w, w]]))
        if relief is not None:
            pr = _patches(relief, fr, fc, w).astype(np.float32)
            pr[pr == 0] = np.nan                                     # 0 = nodata в u8-рельефе bot-9
            ff = [pr[:, :, w, w], _win_mean(pr, 2), _win_mean(pr, 2) - _ring_mean(pr, 5),
                  np.nanstd(pr[:, :, w - 5:w + 6, w - 5:w + 6], axis=(-1, -2))]
            rel_f.append(np.concatenate(ff, 1))
            vf = np.isfinite(pr[:, 0]).mean((1, 2))
            rel_f[-1] = np.column_stack([rel_f[-1], vf])
    _w.__exit__(None, None, None)
    blocks.append(np.concatenate(hm_f).astype(np.float32) if n else np.zeros((0, 6), np.float32))
    names += ["hm_other_ctr", "hm_mean3", "hm_mean7", "hm_sharp", "hm_width_px", "hm_other_max"]
    if mask is not None:
        blocks.append(np.concatenate(m_f).astype(np.float32) if n else np.zeros((0, 5), np.float32))
        names += ["gor_ctr", "gor_max", "fort_ctr", "fort_max", "arch_ctr"]
    if relief is not None:
        C = relief.shape[0]
        rn = list(relief_names[:C]) + [f"ch{i}" for i in range(len(relief_names), C)]
        blocks.append(np.concatenate(rel_f).astype(np.float32) if n else np.zeros((0, 4 * C + 1), np.float32))
        for kind in ("ctr", "mean5", "contrast", "std11"):
            names += [f"rel_{x}_{kind}" for x in rn]
        names += ["valid_frac"]
    nb, nbn = neighbour_features(t, cfg["nb_radii_m"])
    blocks.append(nb)
    names += nbn
    if gor_polys is not None:
        g, gn = gor_poly_features(t, gor_polys)
        blocks.append(g)
        names += gn
    F = np.concatenate(blocks, 1).astype(np.float32)
    assert F.shape[1] == len(names), (F.shape, len(names))
    return F, names
