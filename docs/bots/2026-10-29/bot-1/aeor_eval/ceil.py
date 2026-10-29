# потолок полноты: доля GT (по markup, классу), в которой лежит центр хоть одной детекции D-FINE (любого класса / того же класса) со score ≥ s
import json, gzip, glob, sys, collections
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
mk, src = sys.argv[1], sys.argv[2]
CL = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
gt = [f for f in json.load(open("gt_t.geojson"))["features"] if f["properties"]["markup_type"] == mk]
dets = collections.defaultdict(list)
for p in glob.glob(f"/workspace/koz3_shared/bot-1/aeor_oof/{src}/*.json.gz"):
    d = json.load(gzip.open(p))
    for x in d["dets"]:
        b = x["b"]; dets[d["region"]].append((Point((b[0]+b[2])/2, (b[1]+b[3])/2), x["s"], x["c"]))
done = set(dets); TREES = {}
for s in (0.05, 0.2, 0.5):
    any_, same, tot = collections.Counter(), collections.Counter(), collections.Counter()
    for f in gt:
        r = f["properties"]["region_name"]
        if r not in done: continue
        c = f["properties"]["class_name"]; g = shape(f["geometry"]); tot[c] += 1
        if r not in TREES: TREES[r] = STRtree([pt for pt, _, _ in dets[r]])
        hit = [(dets[r][i][1], dets[r][i][2]) for i in TREES[r].query(g, predicate="contains") if dets[r][i][1] >= s]
        any_[c] += bool(hit); same[c] += any(k == c for _, k in hit)
    print(src, "s>=", s, {c: f"{same[c]}/{any_[c]}/{tot[c]}" for c in tot})
