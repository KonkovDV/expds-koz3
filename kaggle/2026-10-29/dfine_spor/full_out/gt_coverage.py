# Покрытие GT растрами SpOr (только заголовки tif, без пикселей) — воспроизводит фильтр build_chips/region_inv ядра full.
import json, re, os, sys, collections
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform
from shapely.geometry import shape
from shapely.ops import transform as st
from pyproj import Transformer
SRC = "/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full/koz3-dfine-spor-full.py"
s = open(SRC, encoding="utf-8").read()
FOLDS = json.loads(re.search(r"FOLDS = __import__\('json'\)\.loads\(r'''(.*?)'''\)", s, re.S).group(1))
FOLD = {r: f["fold"] for f in FOLDS["folds"] for r in f["test_regions"]}
RU2CLS = {"курганы_целые": 0, "курганы_поврежденные": 1, "городища": 2, "фортификации": 3, "архитектуры": 4}
D = Path("/workspace/data/train-expds-3")
out = []
for reg in sorted(FOLD):
    rd = D / reg
    if not rd.exists(): continue
    tifs, marks, utm = [], [], None
    for p in sorted(rd.rglob("*")):
        n = p.name; k = str(p)
        if n == "UTM.json": utm = p
        elif "_SpOr_" in k and n.lower().endswith((".tif", ".tiff")): tifs.append(p)
        elif n.endswith(".geojson") and "разметка" in k:
            m = re.search(r"_SpOr_(.+)\.geojson$", n)
            if m and m.group(1) in RU2CLS: marks.append((p, RU2CLS[m.group(1)]))
    if not tifs: continue
    crs = None
    if utm:
        m = re.search(r"EPSG::?(\d+)", json.load(open(utm, encoding="utf-8")).get("crs", ""))
        if m: crs = f"EPSG:{m.group(1)}"
    gts = []
    tifinfo = []
    for t in tifs:
        with rasterio.open(t) as ds:
            c = crs
            if c is None:
                lon, lat = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(*ds.xy(ds.height // 2, ds.width // 2))
                c = f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"
            tr, w, h = calculate_default_transform(ds.crs, c, ds.width, ds.height, *ds.bounds, resolution=0.5)
        tifinfo.append((t.name, c, tr, w, h))
    c0 = tifinfo[0][1]
    for p, cl in marks:
        d = json.load(open(p, encoding="utf-8"))
        nm = ((d.get("crs") or {}).get("properties") or {}).get("name", ""); m = re.search(r"EPSG::?(\d+)", nm)
        trf = Transformer.from_crs(f"EPSG:{m.group(1)}" if m else "OGC:CRS84", c0, always_xy=True).transform
        for f in d.get("features", []):
            g = f.get("geometry")
            if not g or g.get("type") not in ("Polygon", "MultiPolygon"): continue
            g = st(trf, shape(g)); g = g if g.is_valid else g.buffer(0)
            if not g.is_empty: gts.append((g.bounds, cl))
    covered = set(); per_tif = []
    for name, c, tr, w, h in tifinfo:
        inv = ~tr; k = 0
        for i, (b, cl) in enumerate(gts):
            (x0, y1), (x1, y0) = inv * (b[0], b[1]), inv * (b[2], b[3])
            bx = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            if bx[2] > 0 and bx[3] > 0 and bx[0] < w and bx[1] < h: covered.add(i); k += 1
        per_tif.append(k)
    out.append({"region": reg, "fold": FOLD[reg], "n_tifs": len(tifs), "gt": len(gts), "gt_x_tifs": len(gts) * len(tifs),
                "gt_in_some_tif": len(covered), "obj_chip_candidates": sum(min(k, 400) for k in per_tif),
                "cls": dict(collections.Counter(cl for _, cl in gts))})
json.dump(out, open("gt_coverage.json", "w"), ensure_ascii=False, indent=0)
print("regions", len(out))
for f in [4, 2, 1, 0, 3]:
    R = [r for r in out if r["fold"] == f]
    print(f"fold {f}: regions {len(R)} gt {sum(r['gt'] for r in R)} gt_x_tifs {sum(r['gt_x_tifs'] for r in R)} covered {sum(r['gt_in_some_tif'] for r in R)} chip_cand {sum(r['obj_chip_candidates'] for r in R)}")
print("ALL gt", sum(r['gt'] for r in out), "gt_x_tifs", sum(r['gt_x_tifs'] for r in out), "covered", sum(r['gt_in_some_tif'] for r in out), "chip_cand", sum(r['obj_chip_candidates'] for r in out))
print("UNCOVERED", [(r['region'], r['gt'] - r['gt_in_some_tif'], r['gt']) for r in out if r['gt'] != r['gt_in_some_tif']])
