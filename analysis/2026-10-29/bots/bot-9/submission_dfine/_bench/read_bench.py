# Замер CPU-части окна (WarpedVRT → UTM 0,5 м, как Scene.read в dfine_main.py) на box, 1 поток GDAL.
import sys, time, json, numpy as np, rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from pyproj import CRS
p = sys.argv[1]; N = int(sys.argv[2]); res = 0.5; T, S = 640, 480
with rasterio.Env(GDAL_NUM_THREADS=1, GDAL_CACHEMAX=256):
    src = rasterio.open(p)
    lon, lat = rasterio.warp.transform(src.crs, "EPSG:4326", [(src.bounds.left + src.bounds.right) / 2], [(src.bounds.top + src.bounds.bottom) / 2])
    zone = int((lon[0] + 180) // 6) + 1; dst = CRS.from_epsg((32600 if lat[0] >= 0 else 32700) + zone)
    tr, w, h = calculate_default_transform(src.crs, dst, src.width, src.height, *src.bounds, resolution=res)
    src_m = abs(calculate_default_transform(src.crs, dst, src.width, src.height, *src.bounds)[0].a)
    rs = Resampling.average if src_m < res else Resampling.bilinear
    vrt = WarpedVRT(src, crs=dst, transform=tr, width=w, height=h, resampling=rs, nodata=0)
    xs = list(range(0, max(1, w - 160), S)); ys = list(range(0, max(1, h - 160), S)); wins = [(x, y) for y in ys for x in xs]
    sel = wins[:: max(1, len(wins) // N)][:N]; ts = []
    for x, y in sel:
        t = time.perf_counter(); cx1, cy1 = min(w, x + T), min(h, y + T)
        a = vrt.read([1, 2, 3][:src.count], window=Window(x, y, cx1 - x, cy1 - y)); ts.append(time.perf_counter() - t)
    out = {"tif": p.rsplit("/", 1)[-1], "src_gsd_m": round(src_m, 3), "dtype": src.dtypes[0], "utm_px": [w, h], "windows_total": len(wins),
           "measured": len(ts), "resampling": rs.name, "read_s_median": round(float(np.median(ts)), 4), "read_s_mean": round(float(np.mean(ts)), 4),
           "read_s_p90": round(float(np.percentile(ts, 90)), 4)}
    print(json.dumps(out, ensure_ascii=False))
