"""LAS → сетка 0,25 м через Rust (expds_fast.grid_accumulate), побитно равная numpy-пути
(x/y/z через scale·X+offset в float64, z → float32, int64-усечение индексов, bincount-сумма/число
по кускам по 5 M точек или minimum.at).

Используется как замена цикла накопления в построении DTM; остальное (EDT-заполнение, гаусс,
маска valid, PMF) не меняется.
"""
import numpy as np

import expds_fast

CHUNK = 5_000_000
RAW = True   # False — через копии полей X/Y/Z/classification (для сверки)


def _c(a, dt):
    return np.ascontiguousarray(np.asarray(a), dtype=dt)


def _raw_layout(arr):
    """(rec_len, cls_off, cls_mask) для сырых записей или None, если раскладка нестандартная."""
    f = arr.dtype.fields or {}
    if not arr.flags.c_contiguous or any(f.get(k, (None, -1))[1] != o or f[k][0] != np.dtype("<i4")
                                          for k, o in (("X", 0), ("Y", 4), ("Z", 8))):
        return None
    if "raw_classification" in f:                 # форматы 0–5: класс = младшие 5 бит
        return arr.dtype.itemsize, f["raw_classification"][1], 0x1F
    if "classification" in f and f["classification"][0] == np.uint8:   # форматы 6–10
        return arr.dtype.itemsize, f["classification"][1], 0xFF
    return None


def accumulate(path, x0, y1, res, W, H, mode, chunk=CHUNK):
    """mode 0: класс 2, сумма/число; 1: без 7/18, сумма/число; 2: без 7/18, минимум.
    → (zsum float64, cnt int32, zmin float32), каждая длины H*W."""
    import laspy
    n = H * W if mode < 2 else 0
    zsum = np.zeros(n, np.float64)
    cnt = np.zeros(n, np.int32)
    zmin = np.full(H * W if mode == 2 else 0, np.inf, np.float32)
    with laspy.open(path) as r:
        hd = r.header
        sc = tuple(float(v) for v in hd.scales)
        of = tuple(float(v) for v in hd.offsets)
        for pts in r.chunk_iterator(chunk):
            lay = _raw_layout(pts.array) if RAW else None
            if lay is not None:
                expds_fast.grid_accumulate_raw(pts.array.view(np.uint8).reshape(-1), *lay, sc, of,
                                               float(x0), float(y1), float(res), int(W), int(H), int(mode),
                                               zsum, cnt, zmin)
                continue
            expds_fast.grid_accumulate(_c(pts.X, np.int32), _c(pts.Y, np.int32), _c(pts.Z, np.int32),
                                       _c(pts.classification, np.uint8), sc, of,
                                       float(x0), float(y1), float(res), int(W), int(H), int(mode),
                                       zsum, cnt, zmin)
    return zsum, cnt, zmin


def ground_grid(path, x0, y1, res, W, H, use_class, all_ground):
    """→ g (H, W) float32/NaN — то же, что даёт накопление в DTM до PMF/заполнения."""
    if use_class or all_ground:
        zsum, cnt, _ = accumulate(path, x0, y1, res, W, H, 0 if use_class else 1)
        return expds_fast.grid_mean(zsum, cnt).reshape(H, W)
    _, _, zmin = accumulate(path, x0, y1, res, W, H, 2)
    return np.where(np.isfinite(zmin), zmin, np.nan).reshape(H, W)
