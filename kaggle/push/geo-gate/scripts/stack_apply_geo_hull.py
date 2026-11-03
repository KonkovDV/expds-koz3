#!/usr/bin/env python3
"""Apply [post] geo_convex_hull to etgr/terfl object_poly in a stack GeoJSON.

usage: stack_apply_geo_hull.py <stack.geojson> <out.geojson> [--markups etgr,terfl]
Streaming (ijson). Non-matching features pass through unchanged.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE / "vendor"), str(HERE)]

try:
    import ijson
except ImportError:
    sys.exit("need ijson (vendored under package vendor/)")

from shapely.geometry import mapping, shape

ap = argparse.ArgumentParser()
ap.add_argument("stack")
ap.add_argument("out")
ap.add_argument("--markups", default="etgr,terfl")
a = ap.parse_args()
MARKS = {x.strip().lower() for x in a.markups.split(",") if x.strip()}

n_in = n_hull = n_skip = 0
with open(a.out, "w", encoding="utf-8") as o, open(a.stack, "rb") as fh:
    o.write('{"type": "FeatureCollection", "features": [')
    first = True
    for f in ijson.items(fh, "features.item", use_float=True):
        n_in += 1
        p = dict(f.get("properties") or {})
        cls = p.get("class_name")
        mk = str(p.get("markup_type") or "").lower()
        if cls == "object_poly" and mk in MARKS and f.get("geometry"):
            try:
                g = shape(f["geometry"])
                if not g.is_empty:
                    hull = g.convex_hull
                    if hull.geom_type == "Polygon":
                        f = {
                            "type": "Feature",
                            "geometry": mapping(hull),
                            "properties": p,
                        }
                        n_hull += 1
                    else:
                        n_skip += 1
            except Exception:
                n_skip += 1
        o.write(("" if first else ", ") + json.dumps(f, ensure_ascii=False))
        first = False
    o.write("]}")

print(json.dumps({
    "n_in": n_in,
    "n_hull": n_hull,
    "n_skip": n_skip,
    "markups": sorted(MARKS),
    "lever": "L3_geo_convex_hull",
}, ensure_ascii=False))
