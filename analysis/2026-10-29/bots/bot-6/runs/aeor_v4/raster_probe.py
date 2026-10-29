"""bot-6 v4: как dfine_aeor читает растры Ae/Or (запрос bot-9: Or 008_Герите прочитан 1×1 px)."""
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, "/workspace/koz3_shared/bot-6/repo")
import numpy as np, rasterio
from expds import dfine_aeor as A, sp_dfine as S
st = A.settings({}); 
for p in map(Path, sys.argv[1:]):
    mk = "ae" if "_Ae_" in p.name else "or"
    with rasterio.open(p) as ds:
        info = {"file": str(p).split("train-expds-3/")[1], "driver": ds.driver, "WxH": [ds.width, ds.height],
                "count": ds.count, "dtypes": sorted(set(ds.dtypes)), "crs": str(ds.crs), "res": [round(r, 5) for r in ds.res],
                "overviews_b1": ds.overviews(1) if ds.count else [], "subdatasets": ds.subdatasets,
                "block": ds.block_shapes[0] if ds.count else None, "nodata": ds.nodata,
                "colorinterp": [c.name for c in ds.colorinterp], "size_mb": round(p.stat().st_size / 2**20, 1)}
    crs = S.utm_for(p, None)
    t0 = time.monotonic()
    with rasterio.Env(GDAL_CACHEMAX=256 * 2**20):
        sc = A.Scene(p, crs, float(st["res_m"][mk]))
        T, ov = int(st["tile"][mk]), int(st["overlap"][mk])
        wins = A.windows(sc.W, sc.H, T, ov)
        xs = max(x for x, _ in wins) + T; ys = max(y for _, y in wins) + T
        cx, cy = max(0, sc.W // 2 - T // 2), max(0, sc.H // 2 - T // 2)
        t1 = time.monotonic(); im = sc.read(cx, cy, T); tr = time.monotonic() - t1
        nz = float((im.max(2) > 0).mean())
        info.update({"utm": crs, "src_m": round(sc.src_m, 4), "overview_level": sc.overview_level,
                     "opened_WxH": [sc.src.width, sc.src.height], "scene_WxH_0.5m": [sc.W, sc.H],
                     "scene_res": round(abs(sc.T.a), 4), "tile": T, "windows": len(wins),
                     "grid_cover_px": [xs, ys], "covers": xs >= sc.W and ys >= sc.H and min(x for x, _ in wins) <= 0,
                     "center_window_nonzero": round(nz, 3), "center_window_mean": round(float(im.mean()), 1),
                     "read_s": round(tr, 1), "open_s": round(t1 - t0, 1), "stretch": sc.stretch})
        sc.close()
    info["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print("RASTER " + json.dumps(info, ensure_ascii=False, default=str), flush=True)
