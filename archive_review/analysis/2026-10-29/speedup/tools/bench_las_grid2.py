"""Стадии DTM «фильтр+индексы+bincount+среднее» по кускам 5M (LAZ уже распакован в память):
numpy-путь (как в dtm.build) vs Rust (las_grid_fast). Побитное сравнение g."""
import sys, time, os, json
import numpy as np, laspy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import expds_fast
import las_grid_fast as lg
from test_las_grid_fast import laz, geom

site = sys.argv[1]; p = laz(site); x0, y1, res, W, H = geom(p)
with laspy.open(p) as r:
    hd = r.header; chunks = list(r.chunk_iterator(5_000_000))
sc = tuple(map(float, hd.scales)); of = tuple(map(float, hd.offsets))

def numpy_path():
    zsum = np.zeros(H * W); cnt = np.zeros(H * W, np.int32)
    for pts in chunks:
        x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z, np.float32)
        m = np.asarray(pts.classification) == 2
        x, y, z = x[m], y[m], z[m]
        ix = ((x - x0) / res).astype(np.int64); iy = ((y1 - y) / res).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        k = iy[ok] * W + ix[ok]
        zsum += np.bincount(k, z[ok], minlength=H * W); cnt += np.bincount(k, minlength=H * W).astype(np.int32)
    return np.where(cnt > 0, zsum / np.maximum(cnt, 1), np.nan).astype(np.float32)

def rust_path():
    zsum = np.zeros(H * W); cnt = np.zeros(H * W, np.int32); e = np.zeros(0, np.float32)
    c = lambda a, d: np.ascontiguousarray(np.asarray(a), dtype=d)
    for pts in chunks:
        expds_fast.grid_accumulate_raw(pts.array.view(np.uint8).reshape(-1), *lg._raw_layout(pts.array), sc, of,
                                       float(x0), float(y1), res, W, H, 0, zsum, cnt, e)
    return expds_fast.grid_mean(zsum, cnt)

def best(f, k=3):
    w, cp = [], []
    for _ in range(k):
        t, c0 = time.perf_counter(), time.process_time(); out = f()
        w.append(time.perf_counter() - t); cp.append(time.process_time() - c0)
    return min(w), min(cp), out

wn, cn, a = best(numpy_path); wr, cr, b = best(rust_path)
assert a.tobytes() == b.tobytes()
print(json.dumps({"site": site, "Mpts": hd.point_count / 1e6, "cells_M": H * W / 1e6,
                  "rayon": os.environ.get("RAYON_NUM_THREADS", "all"), "equal_bytes": True,
                  "numpy_wall": round(wn, 3), "numpy_cpu": round(cn, 3),
                  "rust_wall": round(wr, 3), "rust_cpu": round(cr, 3), "x_wall": round(wn / wr, 1),
                  "x_cpu": round(cn / cr, 1)}, ensure_ascii=False))
