#!/usr/bin/env python3
"""exec A: post-hoc MagGr finds variants on a pix S-file (no GT used):
 per-raster calibration of pix conf (by 'source') and union with classic finds (A18) where pix is empty.
usage: calib_union.py PIX_S.geojson A18.geojson OUTDIR PREFIX"""
import json, sys, os, numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import shape
src, a18, od, pre = sys.argv[1:5]
K = ("maggr", "finds_points"); key = lambda f: (f["properties"].get("markup_type"), f["properties"].get("class_name"))
S = json.load(open(src)); A = json.load(open(a18))
rest = [f for f in S["features"] if key(f) != K]
pix = [f for f in S["features"] if key(f) == K]; cls = [f for f in A["features"] if key(f) == K]
_tr = {}
def utm(fs):
    out = np.zeros((len(fs), 2))
    for i, f in enumerate(fs):
        oc = f["properties"].get("original_crs", "EPSG:32637")
        t = _tr.setdefault(oc, Transformer.from_crs("EPSG:3857", oc, always_xy=True))
        c = shape(f["geometry"]).centroid; out[i] = t.transform(c.x, c.y)
    return out
P, C = utm(pix), utm(cls)
src_of = lambda f: f["properties"]["source"]
def clone(f, **kw):
    g = {"type": "Feature", "geometry": f["geometry"], "properties": dict(f["properties"])}; g["properties"].update(kw); return g
def write(name, finds):
    fs = rest + finds
    for i, f in enumerate(fs): f["properties"]["fid"] = i
    json.dump({**{k: v for k, v in S.items() if k != "features"}, "features": fs}, open(f"{od}/C_{pre}{name}.geojson", "w"), ensure_ascii=False)
    print(name, len(finds), flush=True)
pc = np.array([f["properties"]["confidence"] for f in pix]); srcs = np.array([src_of(f) for f in pix])
# 1) per-raster percentile / blends
pct = np.zeros_like(pc); q99 = np.zeros_like(pc)
for s in set(srcs):
    m = srcs == s; v = pc[m]; r = v.argsort().argsort()
    pct[m] = (r + 1) / len(v); q99[m] = np.clip(v / max(np.quantile(v, 0.99), 1e-6), 0, 1)
variants = {"pct": pct, "b50pct": 0.5 * pc + 0.5 * pct, "q99": q99, "sqrtq99": np.sqrt(q99)}
for n, c in variants.items():
    write(n, [clone(f, confidence=round(float(x), 5)) for f, x in zip(pix, c)])
# 2) union with classic where pix empty (classic keeps own threshold_group 'maggr', pix stays 'maggr_pix')
tree = cKDTree(P)
for d in (1.0, 2.0, 3.0):
    for pmin in (0.0, 0.5):
        tr = cKDTree(P[pc >= pmin]) if pmin > 0 else tree
        dd, _ = tr.query(C, k=1)
        add = [clone(cls[i], threshold_group="maggr_cl") for i in np.where(dd > d)[0]]
        write(f"UE_d{d:g}_p{pmin:g}", [clone(f) for f in pix] + add)
# 3) best combo guess: b50pct + union-empty d2
dd, _ = tree.query(C, k=1)
write("b50pct_UE_d2", [clone(f, confidence=round(float(x), 5)) for f, x in zip(pix, variants["b50pct"])] + [clone(cls[i], threshold_group="maggr_cl") for i in np.where(dd > 2.0)[0]])
