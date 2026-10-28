"""Кандидаты стека для верификатора без лидара: ae/sp_or/or × {курганы, городища} на участках вне notes/las_boxes_59.json.
Метка = официальное правило сопоставления (внутренняя точка кандидата в GT того же класса-группы и пересечение ≥50 % меньшей фигуры),
GT того же участка любой модальности. id кандидата = позиционный индекс в stack_arch.geojson (fid в стеке не уникален)."""
import ijson, json, csv, gzip, time, collections
from shapely.geometry import shape
from shapely.strtree import STRtree
t0 = time.time()
STACK, GTF = "/workspace/rerank/stack_arch.geojson", "/workspace/rerank/gt_train_v2.geojson"
LAS = set(json.load(open("/workspace/expds-koz3/notes/las_boxes_59.json")))
F = json.load(open("folds_family_k5.json", encoding="utf-8"))
fam = F["family_of"]; fold_of = {r: f["fold"] for f in F["folds"] for r in f["test_regions"]}
GROUP = {"kurgany_povrezhdennye": "kurgan", "kurgany_tselye": "kurgan", "gorodishcha": "gor"}
MK = {"ae", "sp_or", "or"}
gt = collections.defaultdict(list)
for f in json.load(open(GTF, encoding="utf-8"))["features"]:
    p = f["properties"]; g = GROUP.get(p["class_name"])
    if g and p["region_name"] not in LAS and f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
        gt[(p["region_name"], g)].append(shape(f["geometry"]).buffer(0))
trees = {k: STRtree(v) for k, v in gt.items()}
rows, n, skip_unknown = [], 0, 0
with open(STACK, "rb") as fh:
    for i, f in enumerate(ijson.items(fh, "features.item", use_float=True)):
        p = f["properties"]; g = GROUP.get(p.get("class_name")); r = p.get("region_name")
        if not g or p.get("markup_type") not in MK or r in LAS: continue
        if r not in fam: skip_unknown += 1; continue
        geom = shape(f["geometry"]).buffer(0)
        if geom.is_empty: continue
        rp = geom.representative_point(); c = geom.centroid; x0, y0, x1, y1 = geom.bounds
        lab = 0
        tr = trees.get((r, g))
        if tr is not None:
            for j in tr.query(geom):
                G = gt[(r, g)][j]
                if G.contains(rp) and geom.intersection(G).area >= 0.5 * min(geom.area, G.area):
                    lab = 1; break
        rows.append([i, p.get("fid"), r, fam[r], fold_of[r], p["markup_type"], p["class_name"], g,
                     round(float(p.get("confidence", 0)), 6), p.get("threshold_group") or "", round(c.x, 2), round(c.y, 2),
                     round(max(x1 - x0, y1 - y0), 2), round(geom.area, 2), lab])
cols = ["stack_idx", "fid", "region", "family", "fold", "markup_type", "class_name", "group", "confidence",
        "threshold_group", "cx", "cy", "size_3857", "area_3857", "label"]
with gzip.open("candidates_nolidar.csv.gz", "wt", encoding="utf-8", newline="") as fo:
    w = csv.writer(fo); w.writerow(cols); w.writerows(rows)
ngt = {f"{k[1]}": 0 for k in gt}
for k, v in gt.items(): ngt[k[1]] += len(v)
st = collections.Counter((r[5], r[7], r[14]) for r in rows)
summ = {"n": len(rows), "sec": round(time.time() - t0), "skip_unknown_region": skip_unknown,
        "by_markup_group_label": {f"{a}/{b}/{c}": v for (a, b, c), v in sorted(st.items())}, "gt_nolidar": ngt,
        "regions": len({r[2] for r in rows}), "families": len({r[3] for r in rows}),
        "pos_by_fold": collections.Counter(r[4] for r in rows if r[14]), "n_by_fold": collections.Counter(r[4] for r in rows)}
json.dump(summ, open("candidates_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False))
