"""Время split_merged на большой области (3000 м² при 0,033 м, как карта Судьбищ) и мелкой сетке."""
import time, numpy as np, sys
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
from expds.geophys import GeoParams, split_merged, label_poles
res = 0.033; n = int(62 / res)
yy, xx = np.mgrid[0:n, 0:n] * res
rng = np.random.default_rng(0)
f = np.zeros((n, n), np.float32)
for cx, cy in rng.uniform(5, 57, (12, 2)):
    f += 3 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
lab = (f >= 1.0).astype(np.int32)
print("pixels", lab.sum(), "area m2", round(lab.sum() * res * res, 1))
for mode, h in (("field", 0.1), ("dt", 0.5)):
    t = time.time(); out, k = split_merged(lab, f, res, GeoParams(poly_k=1.0, split=mode, split_h=h, min_area_m2=5.0)); dt = time.time() - t
    t = time.time(); label_poles(out); dp = time.time() - t
    print(mode, "segments", k, "split s", round(dt, 2), "poles s", round(dp, 2))
