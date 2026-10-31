"""Задача 13, скрипт 2 (диагностика): признаки morph_fast (30 сырых) прямо в representative_point GT-курганов
(povr + tselye) на ЦМР 0,5 м (las_grid класса 2 + пирамидальное заполнение — как в morph_fast/morph_li).
Кольцо: r_peak ≥ 1,5 И ring ≥ 0,15 И hmax ≥ 0,10. Выход: CSV (region, cls, gx, gy, признаки)."""
import json, sys, time
from pathlib import Path
import csv
import numpy as np
WT = Path("/workspace/koz3_shared/bot-2/work/wt_ring")
sys.path.insert(0, str(WT)); sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo/materials")
import morph_fast as MF  # noqa
import morph_li as ML  # noqa
from expds import accel  # noqa
from expds.lidar import fill_interior_holes  # noqa
from compute_metrics import _to_utm  # noqa
from shapely.geometry import shape

regions_pref = sys.argv[2].split(",")
out = Path(sys.argv[1])
gt = json.load(open("/workspace/rerank/gt_train_v2.geojson"))["features"]
rows = []
for pref in regions_pref:
    G = [(f["properties"]["class_name"], _to_utm(shape(f["geometry"]), "auto")) for f in gt
         if f["properties"]["region_name"].startswith(pref + "_")
         and f["properties"]["class_name"] in ("kurgany_povrezhdennye", "kurgany_tselye")]
    region = next(f["properties"]["region_name"] for f in gt if f["properties"]["region_name"].startswith(pref + "_"))
    laz = ML.find_laz("/workspace/data/train-expds-3", region)
    rp = [g.representative_point() for _, g in G]
    gx = np.array([p.x for p in rp]); gy = np.array([p.y for p in rp])
    cx, cy, note = ML.align_coords(laz, gx, gy)
    t0 = time.time()
    dem, gi = accel.las_grid(str(laz), MF.RES, 2, 60_000_000)
    dem = fill_interior_holes(dem, method="pyramid")
    F = MF.features(*MF.windows(dem, gi["left"], gi["top"], np.asarray(cx, float), np.asarray(cy, float)))
    del dem
    for k, (cls, g) in enumerate(G):
        rows.append(dict(region=region, cls=cls, gx=gx[k], gy=gy[k], area=g.area, **{n: float(F[k, MF.J[n]]) for n in MF.RAW}))
    print(json.dumps({"region": region, "n": len(G), "coords": note, "res": gi["res"], "t_s": round(time.time() - t0, 1),
                      "maxrss_mb": round(MF.maxrss_mb(), 1)}, ensure_ascii=False), flush=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
