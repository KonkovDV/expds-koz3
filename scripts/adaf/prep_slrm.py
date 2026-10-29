#!/usr/bin/env python3
"""LAZ -> DEM 0.5 m (ground class 2, interior holes filled) -> SLRM r=20 px -> uint8 [0,1]*255.

Own implementation of the ADAF input recipe (Čož et al. 2026; ADAF manual v1.1):
SLRM = DEM - nan-aware mean over a (2r+1)^2 window, r=20 px (=10 m at 0.5 m/px),
linear stretch [-0.5, 0.5] m -> [0, 1], NaN/nodata -> 0.
Output per region: <out>/<region>.npz {slrm_u8, valid (bool), left, top, res, crs}.
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from expds.lidar import fill_interior_holes  # noqa: E402


def slrm(dem: np.ndarray, r: int = 20) -> np.ndarray:
    v = np.isfinite(dem)
    d = np.where(v, dem, 0).astype(np.float64)
    k = 2 * r + 1
    s = uniform_filter(d, k, mode="nearest")
    c = uniform_filter(v.astype(np.float64), k, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    out = (dem - m).astype(np.float32)
    out[~v] = np.nan
    return out


def slrm_lowmem(dem: np.ndarray, r: int = 20, rows: int = 1024) -> np.ndarray:
    """Same as slrm() but float32 and in row blocks with an r-row halo (for very large rasters, RAM budget).
    Identical up to float32 rounding (~1e-4 m << 1/255 m quantisation of the u8 output)."""
    H, W = dem.shape; k = 2 * r + 1; out = np.empty((H, W), np.float32)
    for y0 in range(0, H, rows):
        a, b = max(0, y0 - r), min(H, y0 + rows + r)
        blk = dem[a:b]; v = np.isfinite(blk)
        s = uniform_filter(np.where(v, blk, 0).astype(np.float32), k, mode="nearest")
        c = uniform_filter(v.astype(np.float32), k, mode="nearest")
        # the r-row halo means kept rows [y0, y0+rows) only see real rows (or the true raster edge)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / c
        o = blk - m; o[~v] = np.nan
        out[y0:min(H, y0 + rows)] = o[y0 - a:y0 - a + min(rows, H - y0)]
    return out


def las_grid_1thread(path: str, res: float, ground_class: int = 2, lowmem: bool = False):
    """Mean z of ground points per cell; single-threaded lazrs backend (bot thread budget)."""
    import laspy
    with laspy.open(path, laz_backend=laspy.LazBackend.Lazrs) as f:
        h = f.header
        left = np.floor(h.mins[0] / res) * res
        top = np.ceil(h.maxs[1] / res) * res
        W = int(np.ceil((h.maxs[0] - left) / res)) + 1
        H = int(np.ceil((top - h.mins[1]) / res)) + 1
        sums = np.zeros(W * H, np.float32 if lowmem else np.float64)
        cnt = np.zeros(W * H, np.int32 if lowmem else np.int64)
        for pts in f.chunk_iterator(1_000_000):
            m = np.asarray(pts.classification) == ground_class if ground_class is not None else \
                np.ones(len(pts), bool)
            if not m.any():
                continue
            x = np.asarray(pts.x)[m]; y = np.asarray(pts.y)[m]; z = np.asarray(pts.z)[m]
            c = np.floor((x - left) / res).astype(np.int64); r = np.floor((top - y) / res).astype(np.int64)
            ok = (c >= 0) & (r >= 0) & (c < W) & (r < H)
            idx = r[ok] * W + c[ok]
            if lowmem:  # per-chunk unique cells instead of full-size bincount temporaries
                u, inv = np.unique(idx, return_inverse=True)
                sums[u] += np.bincount(inv, weights=z[ok]).astype(np.float32)
                cnt[u] += np.bincount(inv).astype(np.int32)
                continue
            sums += np.bincount(idx, weights=z[ok], minlength=W * H)
            cnt += np.bincount(idx, minlength=W * H)
    with np.errstate(invalid="ignore", divide="ignore"):
        if lowmem:
            np.divide(sums, cnt, out=sums); dem = sums.reshape(H, W)
        else:
            dem = (sums / cnt).reshape(H, W).astype(np.float32)
    dem[cnt.reshape(H, W) == 0] = np.nan
    npts = int(cnt.sum()); del cnt
    return dem, {"left": float(left), "top": float(top), "res": res, "points": npts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/data/train-expds-3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--lowmem", action="store_true",
                    help="float32 grid + pyramid hole fill + blocked SLRM (regions that failed on RAM/Qhull)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for laz in sorted(glob.glob(f"{a.data}/*/01_*/*.la[sz]")):
        reg = Path(laz).parts[-3]
        if a.only and reg not in a.only:
            continue
        dst = out / f"{reg}.npz"
        if dst.exists():
            continue
        t = time.time()
        crs = json.load(open(Path(a.data) / reg / "UTM.json"))
        crs = crs.get("crs") or crs.get("epsg") or crs
        dem, info = las_grid_1thread(laz, a.res, lowmem=a.lowmem)
        if info["points"] == 0:  # no class 2 (e.g. only class 8): all points, as expds.lidar does
            del dem
            dem, info = las_grid_1thread(laz, a.res, ground_class=None, lowmem=a.lowmem)
            info["all_points"] = True
        dem = fill_interior_holes(dem, method="pyramid" if a.lowmem else "delaunay")
        s = slrm_lowmem(dem) if a.lowmem else slrm(dem)
        del dem
        valid = np.isfinite(s)
        u8 = np.round(np.clip((np.nan_to_num(s, nan=-0.5) + 0.5), 0, 1) * 255).astype(np.uint8)
        u8[~valid] = 0
        np.savez_compressed(dst, slrm_u8=u8, valid=valid, left=info["left"], top=info["top"],
                            res=info["res"], crs=json.dumps(crs))
        print(json.dumps({"region": reg, "shape": u8.shape, "valid_frac": round(float(valid.mean()), 3),
                          "s": round(time.time() - t, 1), "lowmem": a.lowmem,
                          "all_points": info.get("all_points", False)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
