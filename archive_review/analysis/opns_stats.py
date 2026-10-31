"""Открытость и SVF на рабочем шаге 0,5 м: перцентили на фоне и в центрах курганов эталона."""
import json
import sys
import time

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
import numpy as np
from shapely.geometry import shape

from expds import accel
from expds.candidates import to_work_grid
from expds.catalog import read_utm_json, scan_root
from expds.crs import WEB_MERCATOR, to_crs
from expds.lidar import build_dem

NAMES = sys.argv[1:] or ["036_Тимерево", "009_Хальвасборген", "179_Гнёздово_Лесная_1", "021_Новенькое"]
gt = json.load(open(r"F:\expds3\work\gt_train.geojson", encoding="utf-8"))["features"]
regions = {r.name: r for r in scan_root(r"F:\expds3\train-expds-3")}
for name in NAMES:
    reg = regions[name]
    crs = read_utm_json(reg.utm_json)
    d = build_dem(reg.las[0], res=0.1, fill="delaunay", crs=crs)
    z, step = to_work_grid(d.dem, d.res, 0.5)
    t = time.perf_counter()
    o, s = accel.openness_svf(z, step, 20, 16)
    dt = time.perf_counter() - t
    fin = np.isfinite(o)
    q = [1, 5, 50, 95, 99]
    po = np.percentile(o[fin], q)
    ps = np.percentile(s[fin], q)
    cen = []
    for f in gt:
        pr = f["properties"]
        if pr["region_name"] == name and pr["markup_type"] == "li" and pr["class_name"].startswith("kurgany"):
            p = to_crs(shape(f["geometry"]), WEB_MERCATOR, crs).representative_point()
            c = int((p.x - d.left) / step)
            r = int((d.top - p.y) / step)
            if 0 <= r < o.shape[0] and 0 <= c < o.shape[1] and np.isfinite(o[r, c]):
                cen.append((o[r, c], s[r, c]))
    cen = np.array(cen) if cen else np.zeros((0, 2))
    ko = np.percentile(cen[:, 0], [10, 50, 90]) if len(cen) else []
    ks = np.percentile(cen[:, 1], [10, 50, 90]) if len(cen) else []
    print(f"{name} {o.shape[1]}x{o.shape[0]} {dt:.2f}s | opns p{q}={np.round(po, 1)} kurgan p10/50/90={np.round(ko, 1)}"
          f" n={len(cen)} | svf p{q}={np.round(ps, 3)} kurgan={np.round(ks, 3)}", flush=True)
