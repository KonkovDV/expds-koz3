import sys, hashlib, time, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + '/inference/lib')
from expds import lidar
rng = np.random.default_rng(0)
h, w = 2000, 2500
yy, xx = np.mgrid[0:h, 0:w]
dem = (np.sin(xx / 70) * 3 + np.cos(yy / 50) * 2 + rng.normal(0, .05, (h, w))).astype(np.float32)
dem[rng.random((h, w)) < .05] = np.nan
dem[800:860, 900:1000] = np.nan
d = lidar.Dem(dem=dem, left=0.0, top=0.0, res=0.1, crs='EPSG:32637')
t = time.time(); f = lidar.fill_interior_holes(dem.copy()); t1 = time.time() - t
t = time.time(); hs = lidar.hillshade_set(d); t2 = time.time() - t
t = time.time(); rc = lidar.relief_channels(d, 0.25); t3 = time.time() - t
hsh = hashlib.sha1()
for a in [f] + [hs[k] for k in sorted(hs)] + (list(rc) if isinstance(rc, (list, tuple)) else [rc]):
    a = np.asarray(a); hsh.update(str(a.dtype).encode() + str(a.shape).encode()); hsh.update(np.ascontiguousarray(a).tobytes())
print(sys.argv[1].split('/')[-1], hsh.hexdigest(), f"fill={t1:.2f} hs={t2:.2f} relief={t3:.2f}")
