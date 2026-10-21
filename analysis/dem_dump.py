"""DEM участка текущей сборкой expds_core: время заливки и npz для побитовой сверки."""
import sys
import time

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
import numpy as np

from expds import accel
from expds.catalog import read_utm_json, scan_root

region, out = sys.argv[1], sys.argv[2]
reg = next(r for r in scan_root(r"F:\expds3\train-expds-3") if r.name == region)
dem, info = accel.las_grid(reg.las[0], 0.1, 2, 60_000_000)
t = time.perf_counter()
tin = accel.fill_tin(dem)
dt = time.perf_counter() - t
np.save(out, tin)
print(f"{region} fill_tin {dt:.2f}s occupied {int(np.isfinite(dem).sum())}")
if len(sys.argv) > 3:
    ref = np.load(sys.argv[3])
    same = np.array_equal(ref.view(np.uint32), tin.view(np.uint32))
    print("побитово как раньше:", same)
