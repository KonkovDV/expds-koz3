"""Каналы li целого участка: Rust против NumPy, сколько пикселей отличается."""
import sys

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
import numpy as np

from expds import accel
from expds.catalog import read_utm_json, scan_root
from expds.lidar import build_dem, relief_channels

for region in sys.argv[1:] or ["036_Тимерево", "101_Гнёздово_Днепровская"]:
    reg = next(r for r in scan_root(r"F:\expds3\train-expds-3") if r.name == region)
    d = build_dem(reg.las[0], res=0.1, fill="delaunay", crs=read_utm_json(reg.utm_json))
    tags = ("lrm_s", "lrm_l", "slope", "opns", "svf")
    a, _ = relief_channels(d, 0.5, tags[:3])
    a5, _ = relief_channels(d, 0.5, tags)
    saved = accel._rs
    accel._rs = None
    try:
        b5, _ = relief_channels(d, 0.5, tags)
    finally:
        accel._rs = saved
    print(region, "fused vs 5ch first3 equal:", np.array_equal(a, a5[:3]),
          "| rust vs numpy differing px per channel:", [int((a5[i] != b5[i]).sum()) for i in range(5)],
          "of", a5[0].size)
