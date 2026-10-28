"""LAS → сетка 0,25 м (DTM lidar_a): среднее z точек земли или минимум z по ячейкам.

Rust-модуль expds_fast (rust/expds_fast, колесо в rust/expds_fast/wheels, .so в inference/lib)
или NumPy, если модуль не найден. Оба пути дают побайтно один и тот же результат
(tests/test_las_grid_fast.py): x, y = X·scale + offset в float64, z → float32, индекс ячейки —
усечение (x − left)/cell к int64, суммы float64 в порядке точек по кускам STREAM_POINTS
(как `total += np.bincount(...)`), минимум — как `np.minimum.at`.

Режимы ground.rasterise: "class" — только класс 2, среднее; "bare" — все, кроме шума (7, 18),
среднее; "lowest" — все, кроме шума, минимум. EXPDS_FAST=0 — принудительно NumPy.
"""
from __future__ import annotations

import os

import numpy as np

STREAM_POINTS = 5_000_000
NOISE_CLASSES = (7, 18)
MODES = {"class": 0, "bare": 1, "lowest": 2}

_fast = None
if os.environ.get("EXPDS_FAST", "1") != "0":
    try:
        import expds_fast as _fast  # type: ignore
    except ImportError:
        _fast = None


def backend() -> str:
    return "rust expds_fast" if _fast is not None else "numpy"


def _raw_layout(arr):
    """(rec_len, cls_off, cls_mask) сырых записей laspy или None для нестандартной раскладки."""
    f = arr.dtype.fields or {}
    if not arr.flags.c_contiguous or any(f.get(k, (None, -1))[1] != o or f[k][0] != np.dtype("<i4")
                                          for k, o in (("X", 0), ("Y", 4), ("Z", 8))):
        return None
    if "raw_classification" in f:                                   # форматы 0–5: младшие 5 бит
        return arr.dtype.itemsize, f["raw_classification"][1], 0x1F
    if "classification" in f and f["classification"][0] == np.uint8:  # форматы 6–10
        return arr.dtype.itemsize, f["classification"][1], 0xFF
    return None


def _chunk_numpy(blk, left, top, cell, rows, cols, mode, zsum, cnt, zmin):
    cls = np.asarray(blk.classification)
    keep = cls == 2 if mode == 0 else ~np.isin(cls, NOISE_CLASSES)
    x = np.asarray(blk.x)[keep]
    y = np.asarray(blk.y)[keep]
    z = np.asarray(blk.z, np.float32)[keep]
    col = ((x - left) / cell).astype(np.int64)
    row = ((top - y) / cell).astype(np.int64)
    inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    flat = row[inside] * cols + col[inside]
    if mode < 2:
        zsum += np.bincount(flat, z[inside], minlength=rows * cols)
        cnt += np.bincount(flat, minlength=rows * cols).astype(np.int32)
    else:
        np.minimum.at(zmin, flat, z[inside])


def accumulate(path, left, top, cell, rows, cols, mode, chunk=STREAM_POINTS, use_rust=None):
    """mode 0/1/2 (см. MODES) → (zsum float64, cnt int32, zmin float32); неиспользуемые — длины 0."""
    import laspy

    rust = (_fast is not None) if use_rust is None else bool(use_rust and _fast is not None)
    n = rows * cols
    zsum = np.zeros(n if mode < 2 else 0, np.float64)
    cnt = np.zeros(n if mode < 2 else 0, np.int32)
    zmin = np.full(n if mode == 2 else 0, np.inf, np.float32)
    args = (float(left), float(top), float(cell), int(cols), int(rows), int(mode), zsum, cnt, zmin)
    with laspy.open(str(path)) as f:
        sc = tuple(float(v) for v in f.header.scales)
        of = tuple(float(v) for v in f.header.offsets)
        for blk in f.chunk_iterator(chunk):
            if not rust:
                _chunk_numpy(blk, left, top, cell, rows, cols, mode, zsum, cnt, zmin)
                continue
            arr = blk.array
            lay = _raw_layout(arr)
            if lay is not None:
                _fast.grid_accumulate_raw(arr.view(np.uint8).reshape(-1), *lay, sc, of, *args)
            else:
                c = lambda a, dt: np.ascontiguousarray(np.asarray(a), dtype=dt)  # noqa: E731
                _fast.grid_accumulate(c(blk.X, np.int32), c(blk.Y, np.int32), c(blk.Z, np.int32),
                                      c(blk.classification, np.uint8), sc, of, *args)
    return zsum, cnt, zmin


def rasterise(path, left, top, cell, rows, cols, mode, chunk=STREAM_POINTS, use_rust=None):
    """Замена ground.rasterise: → (rows, cols) float32, NaN в пустых ячейках."""
    m = MODES[mode] if isinstance(mode, str) else int(mode)
    zsum, cnt, zmin = accumulate(path, left, top, cell, rows, cols, m, chunk, use_rust)
    if m < 2:
        if _fast is not None and use_rust is not False:
            cellz = _fast.grid_mean(zsum, cnt)
        else:
            cellz = np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan).astype(np.float32)
    else:
        cellz = np.where(np.isfinite(zmin), zmin, np.nan)
    return cellz.reshape(rows, cols)
