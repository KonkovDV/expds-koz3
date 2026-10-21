import sys
import time

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds import accel
from expds.catalog import scan_root

name = sys.argv[1] if len(sys.argv) > 1 else "021_Новенькое"
reg = [r for r in scan_root(r"F:\expds3\train-expds-3") if r.name == name][0]
p = reg.las[0]
print(p.name, p.stat().st_size // 2**20, "MB")
for i in range(3):
    t = time.perf_counter()
    dem, info = accel.las_grid(p, 0.1, 2, 60_000_000)
    print(f"run {i} {time.perf_counter() - t:.2f}s points {info['points_total']} used {info['points_used']}")
