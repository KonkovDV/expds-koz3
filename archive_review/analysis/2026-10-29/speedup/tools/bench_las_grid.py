"""Время накопления сетки (без чтения LAZ): numpy-путь (scaled x/y/z + фильтр + bincount) vs Rust."""
import sys, time, os, json
import numpy as np, laspy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import expds_fast
from test_las_grid_fast import laz, geom

site = sys.argv[1]; mode = int(sys.argv[2]) if len(sys.argv) > 2 else 0
p = laz(site); x0, y1, res, W, H = geom(p)
with laspy.open(p) as r:
    hd = r.header; pts = r.read_points(hd.point_count)
sc = tuple(map(float, hd.scales)); of = tuple(map(float, hd.offsets))
n = len(pts)

def numpy_path():
    zsum = np.zeros(H * W); cnt = np.zeros(H * W, np.int32)
    x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z, np.float32)
    c = np.asarray(pts.classification)
    m = (c == 2) if mode == 0 else ~np.isin(c, (7, 18))
    x, y, z = x[m], y[m], z[m]
    ix = ((x - x0) / res).astype(np.int64); iy = ((y1 - y) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    k = iy[ok] * W + ix[ok]
    zsum += np.bincount(k, z[ok], minlength=H * W); cnt += np.bincount(k, minlength=H * W).astype(np.int32)
    return zsum, cnt

def rust_path():
    zsum = np.zeros(H * W); cnt = np.zeros(H * W, np.int32); zmin = np.full(H * W, np.inf, np.float32)
    c = lambda a, d: np.ascontiguousarray(np.asarray(a), dtype=d)
    expds_fast.grid_accumulate(c(pts.X, np.int32), c(pts.Y, np.int32), c(pts.Z, np.int32),
                               c(pts.classification, np.uint8), sc, of, float(x0), float(y1), res, W, H, mode,
                               zsum, cnt, zmin)
    return zsum, cnt

def best(f, k=5):
    w, cp = [], []
    for _ in range(k):
        t, c0 = time.perf_counter(), time.process_time(); out = f()
        w.append(time.perf_counter() - t); cp.append(time.process_time() - c0)
    return min(w), min(cp), out

wn, cn, a = best(numpy_path); wr, cr, b = best(rust_path)
assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
print(json.dumps({"site": site, "Mpts": n / 1e6, "cells_M": H * W / 1e6, "mode": mode,
                  "rayon": os.environ.get("RAYON_NUM_THREADS", "all"),
                  "numpy_wall": round(wn, 3), "numpy_cpu": round(cn, 3),
                  "rust_wall": round(wr, 3), "rust_cpu": round(cr, 3), "x_wall": round(wn / wr, 1)}))
