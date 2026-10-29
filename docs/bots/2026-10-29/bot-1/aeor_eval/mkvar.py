"""fam_<V> = база (fam_base) + круги r=4 м из OOF D-FINE Ae/Or (вариант C, как prep10 bot-10), только затронутые юниты."""
import sys, json, gzip, glob, math, os, collections, argparse
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo")
from expds.naming import site_family
from expds.open_ideas import mercator_scale
from shapely.geometry import Point
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--classes", default=""); ap.add_argument("--min", type=float, default=0.0)
a = ap.parse_args()
OOF = "/workspace/koz3_shared/bot-1/aeor_oof/"
T = json.load(open("touched.json")); members = T["member_families"]
cand = collections.defaultdict(list); keep_cls = set(c for c in a.classes.split(",") if c)
for src in a.src.split(","):
    mk = src.split("_")[0].lower()
    for p in glob.glob(OOF + src + "/*.json.gz"):
        d = json.load(gzip.open(p, "rt"))
        for r in d["dets"]:
            if keep_cls and r["c"] not in keep_cls: continue
            if r["s"] < a.min: continue
            x0, y0, x1, y1 = r["b"]
            cand[(mk, r["r"], r["c"])].append(((x0 + x1) / 2, (y0 + y1) / 2, float(r["s"]), (x0, y0, x1, y1)))
def cover(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])); return w * h / sm if sm > 0 else 0.0
CELL = 100.0; MERGE = 4.0; R = 4.0
def cells_of(bx, pad):
    return {(i, j) for i in range(int(math.floor((bx[0] - pad) / CELL)), int(math.floor((bx[2] + pad) / CELL)) + 1)
            for j in range(int(math.floor((bx[1] - pad) / CELL)), int(math.floor((bx[3] + pad) / CELL)) + 1)}
extra = collections.defaultdict(list); stat = collections.Counter()
for (mk, reg, cls), items in sorted(cand.items()):
    items.sort(key=lambda c: -c[2]); grid = collections.defaultdict(list); clusters = []
    for c in items:
        x, y, sc, bx = c; hit = None; seen = set()
        for cell in cells_of(bx, MERGE * mercator_scale(y)):
            for k in grid.get(cell, ()):
                if k in seen: continue
                seen.add(k)
                for m in clusters[k]:
                    if math.hypot(x - m[0], y - m[1]) / mercator_scale((y + m[1]) / 2) <= MERGE or cover(bx, m[3]) >= 0.5:
                        hit = k if hit is None else min(hit, k); break
        if hit is None: clusters.append([c]); hit = len(clusters) - 1
        else: clusters[hit].append(c)
        for cell in cells_of(bx, 0):
            if hit not in grid[cell]: grid[cell].append(hit)
    for cl in clusters:
        b = cl[0]; stat[(mk, cls)] += 1
        geo = Point(b[0], b[1]).buffer(R * mercator_scale(b[1]), quad_segs=4)
        extra[site_family(reg)].append(json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[round(u, 2), round(v, 2)] for u, v in geo.exterior.coords]]},
            "properties": {"class_name": cls, "confidence": round(b[2], 6), "region_name": reg, "crs": "EPSG:3857", "markup_type": mk,
                           "threshold_group": f"{mk}_dfine", "views": len(cl)}}, ensure_ascii=False))
d = f"fam_{a.name}"; os.makedirs(d, exist_ok=True)
for m in members:
    dst = f"{d}/{m}.jsonl"
    if os.path.lexists(dst): os.unlink(dst)
    src = f"fam_base/{m}.jsonl"
    if not extra.get(m):
        if os.path.exists(src): os.symlink(os.path.abspath(src), dst)
        continue
    with open(dst, "w", encoding="utf-8") as fh:
        if os.path.exists(src):
            for l in open(src, encoding="utf-8"): fh.write(l if l.endswith("\n") else l + "\n")
        for l in extra[m]: fh.write(l + "\n")
print(json.dumps({f"{k[0]}|{k[1]}": v for k, v in stat.items()}, ensure_ascii=False), "families", sorted(extra))
