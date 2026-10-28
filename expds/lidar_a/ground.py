"""LAS/LAZ → bare-earth elevation grid (spec §1): ground selection, rasterisation, morphological filter, hole fill."""
from __future__ import annotations

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
    """Stream the cloud into per-cell mean ('class'/'bare') or minimum ('lowest') elevation; NaN where empty."""
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


def fill_holes(cellz, cell):
    """Nearest-ground fill + σ=1 blur of filled cells; valid = within FILL_LIMIT_M of ground (spec §1.5)."""
    holes = np.isnan(cellz)
    dist, near = ndimage.distance_transform_edt(holes, return_indices=True)
    dtm = cellz[near[0], near[1]]
    if holes.any():
        dtm = np.where(holes, ndimage.gaussian_filter(dtm, 1.0), dtm)
    return dtm, dist * cell <= FILL_LIMIT_M


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
