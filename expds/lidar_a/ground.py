"""LAS/LAZ → bare-earth elevation grid (spec §1): ground selection, rasterisation, morphological filter, hole fill."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

BASE_CELL = 0.25          # m
CELL_LIMIT = 250_000_000  # grid cells before the cell size is doubled
FILL_LIMIT_M = 3.0        # filled cells farther than this from ground are invalid
PROBE_POINTS = 2_000_000  # first block used to choose the ground mode
STREAM_POINTS = 5_000_000 # streaming block (fixes the float64 summation order)
NOISE_CLASSES = (7, 18)


@dataclass(frozen=True)
class Grid:
    left: float
    top: float
    cell: float
    rows: int
    cols: int

    @property
    def origin(self):
        return float(self.left), float(self.top), float(self.cell)


def header_box(path):
    """→ (mins, maxs, point_count) from the LAS header."""
    import laspy
    with laspy.open(str(path)) as f:
        return f.header.mins, f.header.maxs, int(f.header.point_count)


def cell_size(mins, maxs, cell=BASE_CELL):
    ex, ey = maxs[0] - mins[0], maxs[1] - mins[1]
    while (ex / cell) * (ey / cell) > CELL_LIMIT:
        cell *= 2
    return cell


def grid_of(mins, maxs, cell=BASE_CELL) -> Grid:
    cell = cell_size(mins, maxs, cell)
    left, top = np.floor(mins[0]), np.ceil(maxs[1])
    cols = int(np.ceil((maxs[0] - left) / cell)) + 1
    rows = int(np.ceil((top - mins[1]) / cell)) + 1
    return Grid(left, top, cell, rows, cols)


def _first_block(path):
    import laspy
    with laspy.open(str(path)) as f:
        for blk in f.chunk_iterator(PROBE_POINTS):
            return blk
    return None


def _bare_earth(blk, square=1.0, spread_limit=0.6):
    """Bare-earth test on one block: 95th percentile of per-1 m-square z spread (squares with ≥ 3 points)."""
    x, y, z = np.asarray(blk.x), np.asarray(blk.y), np.asarray(blk.z)
    cx = ((x - x.min()) // square).astype(np.int64)
    cy = ((y - y.min()) // square).astype(np.int64)
    _, inv, n = np.unique(np.stack([cx, cy]), axis=1, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    hi = np.full(n.size, -np.inf); lo = np.full(n.size, np.inf)
    np.maximum.at(hi, inv, z); np.minimum.at(lo, inv, z)
    spread = (hi - lo)[n >= 3]
    return spread.size > 0 and float(np.percentile(spread, 95)) < spread_limit


def ground_mode(path):
    """'class' (class 2 ≥ 5 % of the first block), 'bare' (cloud is already bare earth) or 'lowest'."""
    blk = _first_block(path)
    if blk is None:
        return "lowest"
    share = float((np.asarray(blk.classification) == 2).mean())
    if share >= 0.05:
        return "class"
    return "bare" if _bare_earth(blk) else "lowest"


def rasterise(path, grid: Grid, mode):
    """Per-cell mean ('class'/'bare') or minimum ('lowest') elevation; NaN where empty.

    Rust expds_fast via expds.las_grid_fast when the module is importable and EXPDS_FAST != 0 (byte-equal to
    rasterise_numpy: tests/test_las_grid_fast.py, tests/test_lidar_a.py); otherwise rasterise_numpy."""
    if os.environ.get("EXPDS_FAST", "1") != "0":
        try:
            from expds import las_grid_fast
        except ImportError:
            las_grid_fast = None
        if las_grid_fast is not None and las_grid_fast._fast is not None:
            return las_grid_fast.rasterise(path, grid.left, grid.top, grid.cell, grid.rows, grid.cols, mode,
                                           chunk=STREAM_POINTS)
    return rasterise_numpy(path, grid, mode)


def rasterise_numpy(path, grid: Grid, mode):
    """Spec §1 reference: stream the cloud into per-cell mean or minimum elevation (float64 sums in point order)."""
    import laspy
    n_cells = grid.rows * grid.cols
    mean_mode = mode in ("class", "bare")
    if mean_mode:
        total = np.zeros(n_cells, np.float64); count = np.zeros(n_cells, np.int32)
    else:
        lowest = np.full(n_cells, np.inf, np.float32)
    with laspy.open(str(path)) as f:
        for blk in f.chunk_iterator(STREAM_POINTS):
            cls = np.asarray(blk.classification)
            keep = cls == 2 if mode == "class" else ~np.isin(cls, NOISE_CLASSES)
            x = np.asarray(blk.x)[keep]; y = np.asarray(blk.y)[keep]
            z = np.asarray(blk.z, np.float32)[keep]
            col = ((x - grid.left) / grid.cell).astype(np.int64)
            row = ((grid.top - y) / grid.cell).astype(np.int64)
            inside = (col >= 0) & (col < grid.cols) & (row >= 0) & (row < grid.rows)
            flat = row[inside] * grid.cols + col[inside]
            if mean_mode:
                total += np.bincount(flat, z[inside], minlength=n_cells)
                count += np.bincount(flat, minlength=n_cells).astype(np.int32)
            else:
                np.minimum.at(lowest, flat, z[inside])
    if mean_mode:
        cellz = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
    else:
        cellz = np.where(np.isfinite(lowest), lowest, np.nan)
    return cellz.reshape(grid.rows, grid.cols)


def morph_ground(lowest, cell, windows_m=(1, 2, 4, 8, 16), dh_start=0.3, dh_slope=0.3, dh_cap=3.0):
    """Progressive morphological filter (Zhang et al. 2003): cells that are not ground → NaN."""
    empty = np.isnan(lowest)
    if empty.all():
        return lowest.copy()
    near = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    surface = lowest[near[0], near[1]]
    is_ground = ~empty
    last = None
    for size in windows_m:
        side = max(3, int(round(size / cell)) | 1)
        opened = ndimage.grey_opening(surface, size=(side, side))
        limit = dh_start if last is None else min(dh_cap, dh_start + dh_slope * (size - last))
        is_ground &= (surface - opened) <= limit
        surface, last = opened, size
    result = lowest.copy()
    result[~is_ground] = np.nan
    return result


# Hole-fill modes (item (3)). "exact": the spec §1.5 procedure as written. "exact_fast" (default): indices-only EDT,
# validity on the integer squared distance, gather + σ=1 blur in row blocks on threads — bit-equal to "exact".
# "band" (opt-in): tiled EDT within BAND_M of ground, nearest coarse block beyond — the relief stack (zero on invalid
# pixels) is equal in value; BAND_M = 80 m covers every relief receptive field from a valid pixel (widest: LRM25 =
# avg-pool 8 + 19-tap blur at 4 m + bilinear ≈ 89 px = 44.5 m; openness 10 m; fill blur 1 m; validity 3 m).
FILL_MODE = os.environ.get("LIDAR_A_FILL", "exact_fast")
BAND_M = float(os.environ.get("LIDAR_A_BAND_M", "80"))
TILE_PX = int(os.environ.get("LIDAR_A_TILE_PX", "4096"))   # band EDT tile; window = tile + 2·band
ROW_BLOCK = 1024
FILL_THREADS = int(os.environ.get("LIDAR_A_FILL_THREADS", os.environ.get("EXPDS_THREADS", "2")))


def fill_holes(cellz, cell, mode=None):
    """Nearest-ground fill + σ=1 blur of filled cells; valid = within FILL_LIMIT_M of ground (spec §1.5, §1a, §1b)."""
    mode = mode or FILL_MODE
    holes = np.isnan(cellz)
    if mode == "exact":
        dist, near = ndimage.distance_transform_edt(holes, return_indices=True)
        dtm = cellz[near[0], near[1]]
        if holes.any():
            dtm = np.where(holes, ndimage.gaussian_filter(dtm, 1.0), dtm)
        return dtm, dist * cell <= FILL_LIMIT_M
    if mode == "exact_fast":
        iy_, ix_ = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
        valid = np.empty(holes.shape, bool)
        for y in range(0, holes.shape[0], ROW_BLOCK):
            valid[y:y + ROW_BLOCK] = _valid_from_indices(iy_[y:y + ROW_BLOCK], ix_[y:y + ROW_BLOCK], cell, y0=y)[0]
    elif mode == "band":
        iy_, ix_, valid = _band_indices(~holes, cell)
    else:
        raise ValueError(f"fill mode {mode!r}")
    return _gather_smooth(cellz, iy_, ix_, holes), valid


def _gather_smooth(g, iy_, ix_, filled, rows=ROW_BLOCK, threads=None):
    """Gather + σ=1 blur of filled cells as in mode "exact", in row blocks on threads.

    gaussian_filter(σ=1, truncate=4) reaches 4 rows; each block is filtered with a 4-row halo, so the separable passes
    do the same float operations for every kept row (reflect only at the true image edges) — byte-equal."""
    from concurrent.futures import ThreadPoolExecutor
    H = g.shape[0]
    dtm = g[iy_, ix_] if H <= rows else np.empty(iy_.shape, g.dtype)
    if H <= rows:
        return np.where(filled, ndimage.gaussian_filter(dtm, 1.0), dtm) if filled.any() else dtm
    R = 4                                 # = int(truncate · σ + 0.5)

    def gather(y):
        dtm[y:y + rows] = g[iy_[y:y + rows], ix_[y:y + rows]]
    ys = list(range(0, H, rows))
    with ThreadPoolExecutor(max(1, threads or FILL_THREADS)) as ex:
        list(ex.map(gather, ys))
        if not filled.any():
            return dtm
        out = np.empty_like(dtm)

        def smooth(y):
            ya, yb = max(0, y - R), min(H, y + rows + R)
            sm = ndimage.gaussian_filter(dtm[ya:yb], 1.0)[y - ya:y - ya + min(rows, H - y)]
            out[y:y + rows] = np.where(filled[y:y + rows], sm, dtm[y:y + rows])
        list(ex.map(smooth, ys))
    return out


def _d2_max(res):
    """Largest integer d2 with sqrt(d2)·res ≤ FILL_LIMIT_M in float64 — the "exact" test (EDT distance · res ≤ FILL_LIMIT_M) with
    scipy's distance = sqrt(dy² + dx²) is monotone in the integer d2, so `d2 <= _d2_max(res)` is the same mask."""
    t = int((FILL_LIMIT_M / res) ** 2) + 2
    while t >= 0 and not (np.sqrt(np.float64(t)) * res <= FILL_LIMIT_M):
        t -= 1
    assert not (np.sqrt(np.float64(t + 1)) * res <= FILL_LIMIT_M)
    return t


def _valid_from_indices(iy_, ix_, res, y0=0, x0=0):
    """→ (valid as in mode "exact", integer d2); int32 math (|dy|, |dx| < 46341 within a row block / window)."""
    H, W = iy_.shape
    dy = iy_ - np.arange(y0, y0 + H, dtype=np.int32)[:, None]
    dx = ix_ - np.arange(x0, x0 + W, dtype=np.int32)[None, :]
    if max(H, W) > 46340 or (dx.size and max(abs(int(dx.min())), abs(int(dx.max()))) > 46340):
        dy, dx = dy.astype(np.int64), dx.astype(np.int64)
    d2 = dy * dy; d2 += dx * dx
    return d2 <= _d2_max(res), d2


def _band_indices(has, res, band_m=None, tile=None, threads=None):
    """Tiled nearest-ground indices: exact within band_m of ground, nearest coarse block beyond."""
    from concurrent.futures import ThreadPoolExecutor
    band = max(int(np.ceil((band_m or BAND_M) / res)), int(np.ceil(FILL_LIMIT_M / res)) + 2)
    tile = tile or TILE_PX
    B = 32
    tile = max(B, tile // B * B)
    H, W = has.shape
    iy_ = np.empty((H, W), np.int32); ix_ = np.empty((H, W), np.int32)
    valid = np.zeros((H, W), bool)
    CY, CX = _coarse_nearest(has, B)

    def coarse(y0, y1, x0, x1):
        cy = np.repeat(np.repeat(CY[y0 // B:-(-y1 // B), x0 // B:-(-x1 // B)], B, 0), B, 1)[:y1 - y0, :x1 - x0]
        cx = np.repeat(np.repeat(CX[y0 // B:-(-y1 // B), x0 // B:-(-x1 // B)], B, 0), B, 1)[:y1 - y0, :x1 - x0]
        return cy, cx

    def one(t):
        y0, x0 = t
        y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
        ya, xa, yb, xb = max(0, y0 - band), max(0, x0 - band), min(H, y1 + band), min(W, x1 + band)
        hw = has[ya:yb, xa:xb]
        if not hw.any():
            iy_[y0:y1, x0:x1], ix_[y0:y1, x0:x1] = coarse(y0, y1, x0, x1)
            return
        jy, jx = ndimage.distance_transform_edt(~hw, return_distances=False, return_indices=True)
        jy = jy[y0 - ya:y1 - ya, x0 - xa:x1 - xa]; jx = jx[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
        jy += ya; jx += xa
        v, d2 = _valid_from_indices(jy, jx, res, y0, x0)
        far = d2 > band * band          # nearest ground may lie outside the window: not exact → coarse fallback
        if far.any():
            cy, cx = coarse(y0, y1, x0, x1)
            jy = np.where(far, cy, jy); jx = np.where(far, cx, jx)
        iy_[y0:y1, x0:x1] = jy; ix_[y0:y1, x0:x1] = jx; valid[y0:y1, x0:x1] = v   # far ⇒ d2 > band² ⇒ not valid
    tiles = [(y, x) for y in range(0, H, tile) for x in range(0, W, tile)]
    with ThreadPoolExecutor(max(1, threads or FILL_THREADS)) as ex:   # scipy's EDT and numpy ufuncs release the GIL
        list(ex.map(one, tiles))
    return iy_, ix_, valid


def _coarse_nearest(has, B):
    """Per B×B block: indices of a ground cell (first in the block) of the nearest ground-bearing block (Hb, Wb)."""
    H, W = has.shape
    Hb, Wb = -(-H // B), -(-W // B)
    hp = np.zeros((Hb * B, Wb * B), bool); hp[:H, :W] = has
    blk = hp.reshape(Hb, B, Wb, B).transpose(0, 2, 1, 3).reshape(Hb, Wb, B * B)
    anyg = blk.any(-1); first = blk.argmax(-1)
    del blk, hp
    if not anyg.any():
        z = np.zeros((Hb, Wb), np.int32)
        return z, z
    by, bx = ndimage.distance_transform_edt(~anyg, return_distances=False, return_indices=True)
    k = first[by, bx]
    return (by * B + k // B).astype(np.int32), (bx * B + k % B).astype(np.int32)


def make_dtm(path, cell=BASE_CELL):
    """→ dtm float32 (rows, cols) without NaN, valid bool, origin (left, top, cell)."""
    mins, maxs, _ = header_box(path)
    grid = grid_of(mins, maxs, cell)
    mode = ground_mode(path)
    cellz = rasterise(path, grid, mode)
    if mode == "lowest":
        cellz = morph_ground(cellz, grid.cell)
    dtm, valid = fill_holes(cellz, grid.cell)
    return dtm.astype(np.float32), valid, grid.origin
