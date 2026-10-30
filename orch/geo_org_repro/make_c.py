"""Build C variants: A with maggr (finds/poly) replaced by organizer OOF in regions the organizer covers; etgr likewise."""
import json, sys
from pathlib import Path
O = Path("/workspace/koz3/org_repro/out")
def load(p): return json.load(open(p))["features"]
def build(A, org, mt, classes, regs, out):
    a = load(A)
    keep = [f for f in a if not (f["properties"]["markup_type"] == mt and f["properties"]["class_name"] in classes
                                 and f["properties"]["region_name"][:3] in regs)]
    add = [f for f in org if f["properties"]["class_name"] in classes and f["properties"]["region_name"][:3] in regs]
    json.dump({"type": "FeatureCollection", "features": keep + add}, open(out, "w"), ensure_ascii=False)
    print(out.name, len(a), "->", len(keep) + len(add))
if __name__ == "__main__":
    A, tag, mt, src = sys.argv[1:5]
    org = load(src)
    allregs = {f["properties"]["region_name"][:3] for f in org}
    no211 = allregs - {"211"}
    for cname, classes in (("all", {"finds_points", "object_poly"}), ("pts", {"finds_points"}), ("poly", {"object_poly"})):
        if mt == "etgr" and cname != "poly": continue
        build(A, org, mt, classes, allregs, O / f"C_{tag}_{cname}.geojson")
        if mt == "maggr": build(A, org, mt, classes, no211, O / f"C_{tag}_{cname}_keep211.geojson")
