import os, sys, time, glob
import numpy as np, rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
path = glob.glob("sample_heavy/*/**/*.tif", recursive=True)[0]
thr = sys.argv[1]
with rasterio.Env(GDAL_NUM_THREADS=thr):
    with rasterio.open(path) as ds:
        print(ds.profile.get("compress"), ds.block_shapes[0], ds.width, ds.height, end=" ")
        t0 = time.perf_counter(); s = 0
        for scale in (1.0, 1.5):
            for (x0, y0) in [(0, 0), (3840, 3840), (7680, 0)]:
                win = Window(x0 * scale, y0 * scale, 4096 * scale, 4096 * scale)
                a = ds.read([1], window=win, out_shape=(1, 4096, 4096),
                            resampling=Resampling.average if scale > 1 else Resampling.bilinear)
                s += int(a.astype(np.int64).sum())
        print(f"GDAL_NUM_THREADS={thr}: {time.perf_counter() - t0:.2f}s checksum={s}")
