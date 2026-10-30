"""NST variant files (prereg/C5B_NST_PREREG.md): m0/m1/m2 on MRP, with and without ORG3. usage: make_nst.py SPLIT OUTDIR"""
import sys, json, math, numpy as np
from pathlib import Path
from scipy.stats import rankdata
import shapely.geometry as sg
K = "/workspace/koz3"; SPLIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
B = Path(K) / "orch/bot-W3"; RPC = ("gorodishcha", "arkhitektury", "object_poly", "fortifikatsii"); ETR = ("212", "213", "214", "215", "216", "217")
def disk(g):
    c = g.representative_point(); sc = max(math.cos(math.atan(math.sinh(c.y / 6378137.0))), 1e-6)
    return sg.mapping(c.buffer(0.05 / sc, quad_segs=8))
c5 = {}
for i, line in zip([int(x) for x in open(B / "variants/C5/C5B.drop").read().split()], open(B / "variants/C5/C5B.jsonl", encoding="utf-8")):
    c5[i] = json.loads(line)["feat"]["properties"]["confidence"]
rows = {}
for p in sorted(SPLIT.glob("*.tsv")):
    for line in open(p, encoding="utf-8"):
        if not any(f'"{c}"' in line for c in RPC): continue
        i, _, rest = line.partition("\t"); f = json.loads(rest)
        if f["properties"].get("class_name") in RPC: rows[int(i)] = (p.stem, f)
allr = sorted(rows)
conf = {i: float(rows[i][1]["properties"]["confidence"]) for i in allr}
rk = {}
for c in RPC:
    ii = [i for i in allr if rows[i][1]["properties"]["class_name"] == c]
    r1 = rankdata([conf[i] for i in ii]) / len(ii); r2 = rankdata([c5.get(i, conf[i]) for i in ii]) / len(ii)
    for i, a, b in zip(ii, r1, r2): rk[i] = 0.5 * (a + b)
isorg = lambda f: f["properties"]["class_name"] == "object_poly" and f["properties"].get("markup_type") == "etgr" and f["properties"]["region_name"][:3] in ETR
org_drop = {i for i in allr if isorg(rows[i][1])}
org_add = [f for f in json.load(open(f"{K}/w4geo/eval/C19etonly_orget_s0.geojson"))["features"] if isorg(f)]
def feat(i, mode):
    fam, f = rows[i]; g = json.loads(json.dumps(f)); g["geometry"] = disk(sg.shape(f["geometry"]))
    if mode == 1 and i in c5: g["properties"]["confidence"] = c5[i]
    if mode == 2: g["properties"]["confidence"] = round(float(rk[i]), 7)
    return fam, g
def orgf():
    out = []
    for f in org_add:
        g = json.loads(json.dumps(f)); g["geometry"] = disk(sg.shape(f["geometry"])); g["properties"].setdefault("crs", "EPSG:3857"); out.append(("Гнёздово", g))
    return out
for mode in (0, 1, 2):
    for org in (False, True):
        nm = f"m{mode}" + ("_O" if org else ""); keep = [i for i in allr if not (org and i in org_drop)]
        items = [feat(i, mode) for i in keep] + (orgf() if org else [])
        with open(OUT / f"{nm}.jsonl", "w", encoding="utf-8") as h:
            for fam, g in items: h.write(json.dumps({"fam": fam, "feat": g}, ensure_ascii=False) + "\n")
        open(OUT / f"{nm}.drop", "w").write("\n".join(map(str, allr)) + "\n"); print(nm, len(items), flush=True)
