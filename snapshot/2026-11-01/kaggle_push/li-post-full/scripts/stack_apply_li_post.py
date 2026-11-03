"""Apply [li_post] class_tau (+ optional self_nms) to li* features in a stack GeoJSON.

Streaming read (ijson); for self_nms we buffer li feats then merge (non-li streamed first).
usage:
  stack_apply_li_post.py <stack.geojson> <out.geojson> [--class-tau] [--nms-m 8]
                         [--floors '{"kurgany_povrezhdennye":0.3,...}']
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# vendor on PYTHONPATH or sibling
HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE / "vendor"), str(HERE)]

try:
    import ijson
except ImportError:
    sys.exit("need ijson (vendored under package vendor/)")

from expds import li_post as LP

ap = argparse.ArgumentParser()
ap.add_argument("stack")
ap.add_argument("out")
ap.add_argument("--class-tau", action="store_true")
ap.add_argument("--nms-m", type=float, default=0.0, help="self_nms radius; 0=off")
ap.add_argument("--floors", default="", help="JSON dict of class→floor; empty=DEFAULT")
a = ap.parse_args()

floors = dict(LP.DEFAULT_CLASS_TAU_FLOOR)
if a.floors:
    floors = {str(k): float(v) for k, v in json.loads(a.floors).items()}

# Pass 1: stream — keep non-li; buffer li as raw feature dicts (geom as geojson)
non_li = []
li_raw = []
n_in = n_drop_tau = 0
with open(a.stack, "rb") as fh:
    for f in ijson.items(fh, "features.item", use_float=True):
        n_in += 1
        p = f.get("properties") or {}
        # synthetic feat for is_li_feat
        probe = {
            "threshold_group": p.get("threshold_group"),
            "markup_type": p.get("markup_type"),
            "class_name": p.get("class_name"),
            "confidence": p.get("confidence"),
        }
        if not LP.is_li_feat(probe):
            non_li.append(f)
            continue
        if a.class_tau:
            cls = p.get("class_name")
            floor = floors.get(cls)
            if floor is not None and float(p.get("confidence") or 0) < float(floor):
                n_drop_tau += 1
                continue
        li_raw.append(f)

n_li_after_tau = len(li_raw)

# NMS needs shapely geoms
n_drop_nms = 0
if a.nms_m and a.nms_m > 0 and li_raw:
    from shapely.geometry import shape

    feats = []
    for f in li_raw:
        p = dict(f.get("properties") or {})
        try:
            g = shape(f["geometry"])
        except Exception:
            g = None
        feats.append({
            "threshold_group": p.get("threshold_group"),
            "markup_type": p.get("markup_type"),
            "class_name": p.get("class_name"),
            "confidence": float(p.get("confidence") or 0),
            "geom": g,
            "_raw": f,
        })
    kept = LP.self_nms(feats, float(a.nms_m))
    n_drop_nms = n_li_after_tau - sum(1 for x in kept if LP.is_li_feat(x))
    # self_nms returns mixed; take li survivors' _raw
    li_out = [x["_raw"] for x in kept if "_raw" in x]
else:
    li_out = li_raw

with open(a.out, "w", encoding="utf-8") as o:
    o.write('{"type": "FeatureCollection", "features": [')
    first = True
    for f in non_li + li_out:
        o.write(("" if first else ", ") + json.dumps(f, ensure_ascii=False))
        first = False
    o.write("]}")

print(json.dumps({
    "n_in": n_in,
    "n_non_li": len(non_li),
    "n_li_after_tau": n_li_after_tau,
    "n_li_out": len(li_out),
    "n_drop_tau": n_drop_tau,
    "n_drop_nms": n_drop_nms,
    "class_tau": bool(a.class_tau),
    "nms_m": float(a.nms_m or 0),
    "floors": floors,
}, ensure_ascii=False))
