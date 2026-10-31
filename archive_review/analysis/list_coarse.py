"""Самые маленькие LAS, у которых шаг DEM больше 0,1 м."""
import math
import sys

import laspy

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.catalog import scan_root

rows = []
for reg in scan_root(r"F:\expds3\train-expds-3"):
    for las in reg.las:
        with laspy.open(las) as f:
            h = f.header
        w = math.ceil((float(h.maxs[0]) - math.floor(float(h.mins[0]) / 0.1) * 0.1) / 0.1) + 1
        hh = math.ceil((float(h.maxs[1]) - math.floor(float(h.mins[1]) / 0.1) * 0.1) / 0.1) + 1
        factor = math.ceil(math.sqrt(w * hh / 60e6)) if w * hh > 60e6 else 1
        if factor > 1:
            rows.append((las.stat().st_size, factor, w, hh, reg.name, str(las)))
rows.sort()
print(f"крупных {len(rows)}")
for size, factor, w, hh, name, path in rows[:12]:
    print(f"{size / 1e6:7.1f} MB  x{factor}  {w}x{hh}  {name}")
    print(f"    {path}")
