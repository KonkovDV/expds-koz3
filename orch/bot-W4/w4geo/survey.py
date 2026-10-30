import rasterio, glob, os, numpy as np
for p in sorted(glob.glob('/workspace/koz3/w4geo/train/*/*/*.tif')):
    with rasterio.open(p) as d:
        print(os.path.relpath(p,'/workspace/koz3/w4geo/train')[:70], d.count, d.dtypes[0], d.width, d.height, round(abs(d.res[0]),3), (d.crs.to_string() if d.crs else None), d.nodata)
