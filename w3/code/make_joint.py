"""C5B joint (prereg/C5B_JOINT_PREREG.md). usage: make_joint.py SPLIT OUTDIR"""
import sys, json, math
from pathlib import Path
import shapely.geometry as sg
K = "/workspace/koz3"; SPLIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
B = Path(K) / "orch/bot-W3"
RPC = ("gorodishcha", "arkhitektury", "object_poly", "fortifikatsii")
ETR = ("212", "213", "214", "215", "216", "217")
def disk(g):
    c = g.representative_point(); sc = max(math.cos(math.atan(math.sinh(c.y / 6378137.0))), 1e-6)
    return sg.mapping(c.buffer(0.05 / sc, quad_segs=8))
c5 = {}
dr5 = [int(x) for x in open(B / "variants/C5/C5B.drop").read().split()]
for i, line in zip(dr5, open(B / "variants/C5/C5B.jsonl", encoding="utf-8")): c5[i] = json.loads(line)["feat"]["properties"]["confidence"]
rows = {}  # idx -> (fam, feat)
for p in sorted(SPLIT.glob("*.tsv")):
    for line in open(p, encoding="utf-8"):
        if not any(f'"{c}"' in line for c in RPC): continue
        i, _, rest = line.partition("\t"); f = json.loads(rest)
        if f["properties"].get("class_name") in RPC: rows[int(i)] = (p.stem, f)
isorg = lambda f: f["properties"]["class_name"] == "object_poly" and f["properties"].get("markup_type") == "etgr" and f["properties"]["region_name"][:3] in ETR
org_drop = {i for i, (_, f) in rows.items() if isorg(f)}
org_add = [f for f in json.load(open(f"{K}/w4geo/eval/C19etonly_orget_s0.geojson"))["features"] if isorg(f)]
for f in org_add: f["properties"].setdefault("crs", "EPSG:3857")
print("rows", len(rows), "org_drop", len(org_drop), "org_add", len(org_add), "c5", len(c5), "c5∩org", len(org_drop & set(c5)), flush=True)
def W(name, items, drop):
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as h:
        for fam, f in items: h.write(json.dumps({"fam": fam, "feat": f}, ensure_ascii=False) + "\n")
    open(OUT / f"{name}.drop", "w").write("\n".join(map(str, sorted(drop))) + "\n")
    print(name, len(items), len(drop), flush=True)
def feat(i, conf5=False, mrp=False):
    fam, f = rows[i]; g = json.loads(json.dumps(f))
    if conf5 and i in c5: g["properties"]["confidence"] = c5[i]
    if mrp: g["geometry"] = disk(sg.shape(f["geometry"]))
    return fam, g
def orgf(mrp):
    out = []
    for f in org_add:
        g = json.loads(json.dumps(f))
        if mrp: g["geometry"] = disk(sg.shape(f["geometry"]))
        out.append(("Гнёздово", g))
    return out
allr = sorted(rows); c5r = sorted(set(c5) - org_drop)
W("MRP", [feat(i, mrp=True) for i in allr], allr)
W("ORG", orgf(False), org_drop)
W("C5Bj", [feat(i, conf5=True) for i in sorted(c5)], sorted(c5))
keep = [i for i in allr if i not in org_drop]
W("S0", [feat(i, mrp=True) for i in keep] + orgf(True), allr)
W("S1", [feat(i, conf5=True, mrp=True) for i in keep] + orgf(True), allr)
