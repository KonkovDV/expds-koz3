#!/usr/bin/env python3
"""exec A: split pix MagGr finds into threshold_groups by test-time raster stats (no GT):
 honest per-fold thresholds then adapt per raster type. usage: raster_groups.py PIX_S.geojson OUTDIR PREFIX"""
import json, sys, os, numpy as np
src, od, pre = sys.argv[1:4]
K = ("maggr", "finds_points"); key = lambda f: (f["properties"].get("markup_type"), f["properties"].get("class_name"))
S = json.load(open(src)); rest = [f for f in S["features"] if key(f) != K]; pix = [f for f in S["features"] if key(f) == K]
pc = np.array([f["properties"]["confidence"] for f in pix]); srcs = np.array([f["properties"]["source"] for f in pix])
us = sorted(set(srcs)); st = {}
for s in us:
    v = pc[srcs == s]; st[s] = dict(q90=np.quantile(v, .9), n=len(v), nhi=(v >= .5).sum(), med=np.median(v))
def write(name, grp):
    fs = rest + [{"type": "Feature", "geometry": f["geometry"], "properties": {**f["properties"], "threshold_group": "maggr_pix_" + grp[f["properties"]["source"]]}} for f in pix]
    for i, f in enumerate(fs): f["properties"]["fid"] = i
    json.dump({**{k: v for k, v in S.items() if k != "features"}, "features": fs}, open(f"{od}/C_{pre}{name}.geojson", "w"), ensure_ascii=False)
    print(name, {g: sum(1 for s in us if grp[s] == g) for g in set(grp.values())}, flush=True)
for stat in (() if os.environ.get("CUTS") else ("q90", "med", "nhi", "n")):
    vals = np.array([st[s][stat] for s in us])
    for nb in (2, 3):
        cuts = np.quantile(vals, np.linspace(0, 1, nb + 1)[1:-1])
        write(f"G{stat}{nb}", {s: str(int(np.searchsorted(cuts, st[s][stat]))) for s in us})
for s in us: print(s[:40], {k: round(float(v), 3) for k, v in st[s].items()})
write("Gtype", {s: ("cv" if "цвет" in s else "gr") for s in us})
import os
for cut in [int(x) for x in os.environ.get("CUTS","300").split(",")]:
    write(f"Gtype_n{cut}", {s: ("cv" if "цвет" in s else "gr") + ("lo" if st[s]["n"] < cut else "hi") for s in us})
    write(f"Gn{cut}", {s: ("lo" if st[s]["n"] < cut else "hi") for s in us})
