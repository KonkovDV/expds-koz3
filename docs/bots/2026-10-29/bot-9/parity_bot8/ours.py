# Наша prep-цепочка (koz3-cnx-li-prep: convnext_main.build_relief) на одном LAS — вендор из ДЕКОДИРОВАННОГО prep-ядра.
import sys, json, time, numpy as np
V = "/workspace/koz3_shared/bot-9/kaggle/convnext_lidar/_scratch/lic/prep_vendor"; sys.path.insert(0, V)
from expds import accel as XA, lidar as XL
from expds.candidates import to_work_grid
import relief_ext as RX
TAGS5 = ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")
t = time.time(); las = sys.argv[1]
d = XL.build_dem(las, res=0.1, fill="delaunay", crs="EPSG:32633")
base, tr = XL.relief_channels(d, 0.5, TAGS5[:3])
z, step = to_work_grid(d.dem, d.res, 0.5)
z = np.where(base[0] > 0, z, np.nan).astype(np.float32)
ext, stats = RX.extra_channels(z, step, d.left, d.top, accel=None, tags=TAGS5[3:])
arr = np.concatenate([base, ext]).astype(np.uint8)
np.savez_compressed("ours.npz", arr=arr, z=z, left=d.left, top=d.top, step=step, dem_res=d.res)
print(json.dumps(dict(backend=XA.backend(), shape=arr.shape, left=d.left, top=d.top, step=step, dem_res=d.res, stats=stats, t=round(time.time()-t,1))))
