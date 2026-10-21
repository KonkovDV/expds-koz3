"""DEM и каналы li одного участка той версией кода, что первой в sys.path; сохраняет в npz."""
import sys

import numpy as np

import expds
import expds_core
from expds.catalog import read_utm_json, scan_root
from expds.lidar import build_dem, relief_channels

region, out = sys.argv[1], sys.argv[2]
reg = next(r for r in scan_root(r"F:\expds3\train-expds-3") if r.name == region)
d = build_dem(reg.las[0], res=0.1, fill="delaunay", crs=read_utm_json(reg.utm_json))
arr, tr = relief_channels(d, 0.5, ("lrm_s", "lrm_l", "slope"))
np.savez(out, dem=d.dem, arr=arr, origin=np.array([tr.c, tr.f, tr.a]))
print(expds.__file__, expds_core.__file__, expds_core.version(), arr.shape)
