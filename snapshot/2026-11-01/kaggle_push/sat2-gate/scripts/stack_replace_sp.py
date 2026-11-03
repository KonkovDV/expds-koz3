"""REPLACE sat layer: drop markup_type in sp_* (kurgan classes by default) + append OOF sp_det2.
usage: stack_replace_sp.py <stack.geojson> <oof.geojson> <out.geojson> [--classes c1,c2] [--min-conf-json thr.json] [--scale 1.0]
Streaming (ijson). conf/geometry of kept non-sat rows unchanged. New feats get threshold_group=sp_det2.
"""
from __future__ import annotations
import argparse, collections, json, sys
try:
    import ijson
except ImportError:
    sys.exit("need ijson")

ap = argparse.ArgumentParser()
ap.add_argument("stack"); ap.add_argument("oof"); ap.add_argument("out")
ap.add_argument("--classes", default="kurgany_povrezhdennye,kurgany_tselye")
ap.add_argument("--scale", type=float, default=1.0, help="sat_thr_relax scale: keep oof if conf >= scale * default_thr[class]")
ap.add_argument("--min-conf-json", default="", help="JSON {class: thr} for scale filter; empty = keep all oof")
ap.add_argument("--drop-prefixes", default="sp_", help="comma prefixes of markup_type to drop when class in --classes")
a = ap.parse_args()
CLS = set(a.classes.split(",")) if a.classes else None
PRE = tuple(x for x in a.drop_prefixes.split(",") if x)

def sel_drop(p):
    if CLS is not None and p.get("class_name") not in CLS:
        return False
    mk = str(p.get("markup_type") or "")
    return any(mk.startswith(x) for x in PRE)

thr = {}
if a.min_conf_json:
    thr = json.load(open(a.min_conf_json, encoding="utf-8"))

def keep_oof(p):
    if not thr:
        return True
    c = p.get("class_name")
    t = thr.get(c)
    if t is None:
        return True
    return float(p.get("confidence") or 0) >= float(t) * float(a.scale)

F = []
for f in json.load(open(a.oof, encoding="utf-8"))["features"]:
    p = f.get("properties") or {}
    if CLS is not None and p.get("class_name") not in CLS:
        continue
    if keep_oof(p):
        F.append(f)

drop = collections.Counter(); kept = 0
with open(a.out, "w", encoding="utf-8") as o, open(a.stack, "rb") as fh:
    o.write('{"type": "FeatureCollection", "features": [')
    first = True
    for f in ijson.items(fh, "features.item", use_float=True):
        p = f.get("properties") or {}
        if sel_drop(p):
            drop[(p.get("threshold_group"), p.get("class_name"), p.get("markup_type"))] += 1
            continue
        o.write(("" if first else ", ") + json.dumps(f, ensure_ascii=False))
        first = False
        kept += 1
    for i, f in enumerate(F):
        p = dict(f["properties"])
        p["threshold_group"] = "sp_det2"
        p.setdefault("markup_type", "sp_or")
        p["fid"] = f"sp_det2_rep_{i}"
        o.write(", " + json.dumps({"type": "Feature", "geometry": f["geometry"], "properties": p}, ensure_ascii=False))
    o.write("]}")
print(json.dumps(dict(
    kept=kept, dropped=sum(drop.values()), added=len(F), scale=a.scale,
    drop={f"{g}|{c}|{m}": n for (g, c, m), n in sorted(drop.items(), key=str)},
), ensure_ascii=False))
