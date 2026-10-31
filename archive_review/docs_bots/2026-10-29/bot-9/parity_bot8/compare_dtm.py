# DTM на 0.5 м: наш z (build_dem 0.1 delaunay → to_work_grid 5x5) vs bot-8 make_dtm 0.25 → avg_pool 2x2 (как terrain._resample).
import sys, json, numpy as np, torch
sys.path.insert(0, "b8_95bc0a6")
from expds.lidar_a import ground, terrain
dtm, valid, org = ground.make_dtm("esbu.las")
z8, v8 = terrain._resample(dtm, valid, org[2], "cpu"); z8 = z8[0, 0].numpy(); v8 = v8[0, 0].numpy() > 0.5
o = np.load("ours.npz"); z = o["z"]; H, W = z.shape
z8 = z8[1:1 + H, 1:1 + W]; v8 = v8[1:1 + H, 1:1 + W]; m = np.isfinite(z) & v8
d = np.abs(z[m] - z8[m]); r = dict(max_abs_m=round(float(d.max()), 4), mean_abs_m=round(float(d.mean()), 4), p99_abs_m=round(float(np.percentile(d, 99)), 4), median_abs_m=round(float(np.median(d)), 4), n=int(m.sum()))
json.dump(r, open("compare_dtm.json", "w"), indent=1); print(r)
