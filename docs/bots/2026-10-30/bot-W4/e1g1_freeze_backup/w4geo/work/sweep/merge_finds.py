#!/usr/bin/env python3
"""exec A: C* = base geojson (e.g. CetO3) with maggr finds_points replaced by those from a finds-variant geojson."""
import json, sys
base, fsrc, out = sys.argv[1:4]
K = ("maggr", "finds_points")
k = lambda f: (f["properties"].get("markup_type"), f["properties"].get("class_name"))
b = json.load(open(base)); s = json.load(open(fsrc))
feats = [f for f in b["features"] if k(f) != K] + [f for f in s["features"] if k(f) == K]
for i, f in enumerate(feats): f["properties"]["fid"] = i
b["features"] = feats; json.dump(b, open(out, "w"), ensure_ascii=False)
print("written", len(feats), out)
