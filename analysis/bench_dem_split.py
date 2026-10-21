"""Разложение времени DEM: чтение сетки и заливка TIN."""
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")

import numpy as np

from expds import accel
from expds.lidar import build_dem, fill_interior_holes

LAS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"F:\expds3\train-expds-3\009_Хальвасборген\01_Хальвасборген_La_точки\01_Хальвасборген_La_точки.las")


def once(label):
    t0 = time.perf_counter()
    dem, info = accel.las_grid(LAS, 0.1, 2, 60_000_000)
    t1 = time.perf_counter()
    filled = fill_interior_holes(dem, method="delaunay")
    t2 = time.perf_counter()
    print(f"{label} grid {t1 - t0:.3f}s  fill {t2 - t1:.3f}s  "
          f"{dem.shape} res {info['res']} finite {np.isfinite(filled).mean():.3f}", flush=True)


print(accel.backend(), flush=True)
once("cold")
once("warm")
