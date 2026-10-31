"""Проверки снимков Ae на коробке: метаданные, обзор (сторона ≤1024), покрытие эталона ae/li.
Запуск: OMP_NUM_THREADS=2 GDAL_NUM_THREADS=2 GDAL_CACHEMAX=128 nice -n 10 /tmp/dfine_venv/bin/python ae_raster_check.py OUT.json"""
import json, sys, glob, os, math, collections
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from pyproj import Transformer
from shapely.geometry import shape
ROOT = "/workspace/data/train-expds-3"
gt = json.load(open("/workspace/koz3_shared/bot-6/work/gt_raw_v3.geojson"))["features"]
pts = collections.defaultdict(list)  # (region, markup) -> [(class, x3857, y3857)]
for f in gt:
    p = f["properties"]
    if p["markup_type"] not in ("ae", "li"):
        continue
    c = shape(f["geometry"]).representative_point()
    pts[(p["region_name"], p["markup_type"])].append((p["class_name"], c.x, c.y))
out = []
for path in sorted(glob.glob(f"{ROOT}/*/03_*_Ae_*/*.tif")):
    reg = os.path.relpath(path, ROOT).split("/")[0]
    utm = json.load(open(f"{ROOT}/{reg}/UTM.json")).get("crs") if os.path.exists(f"{ROOT}/{reg}/UTM.json") else None
    with rasterio.open(path) as ds:
        crs = ds.crs.to_string() if ds.crs else None
        f = max(1, math.ceil(max(ds.width, ds.height) / 1024))
        h, w = max(1, ds.height // f), max(1, ds.width // f)
        a = ds.read(out_shape=(ds.count, h, w), resampling=Resampling.nearest)
        nd = ds.nodata
        valid = np.ones((h, w), bool)
        for b in range(a.shape[0]):
            valid &= (a[b] != (0 if nd is None else nd))
        tr = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
        b = ds.bounds
        cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy)[1] if crs else None
        ground = ds.res[0] * (math.cos(math.radians(lat)) if crs and crs.endswith("3857") else 1.0)
        row = dict(region=reg, file=os.path.basename(path), size_mb=round(os.path.getsize(path) / 1e6, 1),
                   crs=crs, utm_json=utm, res=round(ds.res[0], 4), ground_res_m=round(ground, 4),
                   wh=[ds.width, ds.height], count=ds.count, dtype=ds.dtypes[0], nodata=nd,
                   compress=ds.compression.value if ds.compression else None,
                   colorinterp=[c.name for c in ds.colorinterp], overview_factor=f,
                   n_overviews=len(ds.overviews(1)), lat=round(lat, 2) if lat else None,
                   valid_share=round(float(valid.mean()), 4))
        v = a[:, valid].astype(np.float64) if valid.any() else np.zeros((a.shape[0], 0))
        row["bands_identical"] = bool(a.shape[0] >= 3 and np.array_equal(a[0], a[1]) and np.array_equal(a[1], a[2]))
        if v.size:
            row["p1_p50_p99"] = [float(x) for x in np.percentile(v[0], [1, 50, 99])]
            row["share_ge254"] = round(float((v[0] >= 254).mean()), 4)
            row["share_le1"] = round(float((v[0] <= 1).mean()), 4)
            row["std_valid"] = round(float(v[0].std()), 2)
        t = Transformer.from_crs("EPSG:3857", crs, always_xy=True) if crs else None
        for mk in ("ae", "li"):
            P = pts.get((reg, mk), [])
            n_in = n_val = 0
            kin = collections.Counter()
            for cls, x, y in P:
                X, Y = t.transform(x, y)
                r, c = rowcol(tr, X, Y)
                if 0 <= r < h and 0 <= c < w:
                    n_in += 1
                    if valid[r, c]:
                        n_val += 1
                        kin[cls] += 1
            row[f"gt_{mk}"] = dict(n=len(P), in_bounds=n_in, in_valid=n_val, by_class_valid=dict(kin))
        del a, valid, v
    out.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
import resource; print("maxrss_mb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024, file=sys.stderr)
