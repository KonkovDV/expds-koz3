"""Полигонизация и максимум по меткам: прежний путь против нового на растре 3000×3000."""
import sys
import time

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
import numpy as np
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy.ndimage import label
from shapely.geometry import shape

from expds import accel
from expds.crs import WEB_MERCATOR, to_crs
from expds.raster import mask_to_polygons

rng = np.random.default_rng(0)
p = rng.random((3000, 3000), dtype=np.float32)
from scipy.ndimage import uniform_filter

p = uniform_filter(p, 9)
mask = p > np.quantile(p, 0.9)
lab, n = label(mask)
lab = lab.astype(np.int32)
tr = from_origin(500000, 6100000, 0.5, 0.5)
print("меток", n)

t = time.perf_counter()
ref = np.zeros(n + 1, np.float32)
np.maximum.at(ref, lab.ravel(), p.ravel())
t1 = time.perf_counter()
got = accel.label_max(lab, p, n)
t2 = time.perf_counter()
print(f"maximum.at {t1 - t:.3f} с, label_max {t2 - t1:.3f} с, равны {np.array_equal(ref, got)}")

t = time.perf_counter()
old = [(to_crs(shape(g).simplify(0.25, preserve_topology=True), "EPSG:32636", WEB_MERCATOR), int(v))
       for g, v in shapes(lab, mask=lab > 0, transform=tr, connectivity=8)]
t1 = time.perf_counter()
new = mask_to_polygons(lab, tr, "EPSG:32636", simplify=0.25)
t2 = time.perf_counter()
same = len(old) == len(new) and all(a.equals_exact(b, 0) and va == vb for (a, va), (b, vb) in zip(old, new))
print(f"полигонов {len(new)}: по одному {t1 - t:.2f} с, пачкой {t2 - t1:.2f} с, совпадают {same}")
