#!/usr/bin/env python3
"""Apply sat_recall levers to a stack GeoJSON (post-hoc).

usage:
  stack_apply_sat_recall.py <stack.geojson> <out.geojson>
      [--band-verify] [--agree-accept] [--cross-nms]
      [--band-scale 0.90] [--verify-floor 0.40]
      [--agree-radius 16] [--agree-scale 0.85]
      [--nms-m 4.0] [--default-thr 0.15]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE / "vendor"), str(HERE)]

from shapely.geometry import mapping, shape
from expds import sat_recall as SR

ap = argparse.ArgumentParser()
ap.add_argument("stack")
ap.add_argument("out")
ap.add_argument("--band-verify", action="store_true")
ap.add_argument("--agree-accept", action="store_true")
ap.add_argument("--cross-nms", action="store_true")
ap.add_argument("--band-scale", type=float, default=0.90)
ap.add_argument("--verify-floor", type=float, default=0.40)
ap.add_argument("--agree-radius", type=float, default=16.0)
ap.add_argument("--agree-scale", type=float, default=0.85)
ap.add_argument("--nms-m", type=float, default=4.0)
ap.add_argument("--default-thr", type=float, default=0.15)
ap.add_argument("--no-verify-policy", default="admit",
                help="admit|reject mid-band without verify_p (post-hoc default admit)")
a = ap.parse_args()

if abs(a.agree_radius - 8.0) < 1e-9:
    sys.exit("refuse agree radius_m=8.0 (CLOSED sat 8m cluster)")

raw = json.load(open(a.stack, encoding="utf-8"))
feats = []
for f in raw["features"]:
    p = dict(f.get("properties") or {})
    try:
        g = shape(f["geometry"]) if f.get("geometry") else None
    except Exception:
        g = None
    feats.append({
        "class_name": p.get("class_name"),
        "confidence": float(p.get("confidence") or 0),
        "markup_type": p.get("markup_type"),
        "threshold_group": p.get("threshold_group") or p.get("markup_type"),
        "region_name": p.get("region_name"),
        "geom": g,
        "verify_p": p.get("verify_p"),
        "_raw": f,
        "_props": p,
    })

cfg = {
    "default_threshold": float(a.default_thr),
    "thresholds": {},
    "sat_band_verify": {
        "enabled": bool(a.band_verify),
        "scale": float(a.band_scale),
        "verify_floor": float(a.verify_floor),
        "no_verify_policy": a.no_verify_policy,
    },
    "sat_agree_accept": {
        "enabled": bool(a.agree_accept),
        "radius_m": float(a.agree_radius),
        "scale": float(a.agree_scale),
    },
    "sat_cross_nms": {
        "enabled": bool(a.cross_nms),
        "nms_m": float(a.nms_m),
    },
}

n_in = len(feats)
# If no lever enabled: identity keep all
if not (a.band_verify or a.agree_accept or a.cross_nms):
    out_feats = feats
else:
    out_feats = SR.apply_thresholds(feats, cfg, by_markup=None)

with open(a.out, "w", encoding="utf-8") as o:
    o.write('{"type": "FeatureCollection", "features": [')
    first = True
    for x in out_feats:
        p = dict(x.get("_props") or {})
        p["confidence"] = float(x.get("confidence", p.get("confidence") or 0))
        if x.get("sat_agree"):
            p["sat_agree"] = True
        if x.get("sat_band"):
            p["sat_band"] = True
        geom = mapping(x["geom"]) if x.get("geom") is not None else (x.get("_raw") or {}).get("geometry")
        f = {"type": "Feature", "geometry": geom, "properties": p}
        o.write(("" if first else ", ") + json.dumps(f, ensure_ascii=False))
        first = False
    o.write("]}")

print(json.dumps({
    "n_in": n_in,
    "n_out": len(out_feats),
    "cfg": {k: v for k, v in cfg.items() if k.startswith("sat_") or k == "default_threshold"},
}, ensure_ascii=False))
