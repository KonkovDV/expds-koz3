# Сборка входов для замера D-FINE OOF (фолды 4,2,1) на конвейере бота 10 (только чтение bot-10, запись только сюда).
# fam_base: семьи фолдов 4,2,1 из bot-10/work/fam (симлинки); fam_C: база + круги r=4 м (дубли тайлов склеены как в
# coco_preds_to_stack: участок+класс, центр ≤4 м на земле или покрытие рамок ≥0,5); fam_E: база + эллипсы как есть (buffer(0));
# fam_Conly: только круги D-FINE. gt_421 — эталон семей фолдов 4,2,1; gt_421_sp — его часть sp_or.
import sys, json, re, os, math, collections, resource
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo"))
from expds.naming import site_family
from expds.open_ideas import mercator_scale
from shapely.geometry import shape, Point, mapping
OUT = Path(__file__).resolve().parent; os.chdir(OUT); SRC = Path("/workspace/kaggle_out/koz3-dfine-spor-full")
s = open("/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full/koz3-dfine-spor-full.py", encoding="utf-8").read()
F = json.loads(re.search(r"FOLDS = __import__\('json'\)\.loads\(r'''(.*?)'''\)", s, re.S).group(1))
FAMS = set(); fam_fold = {}
for fd in F["folds"]:
    if fd["fold"] in (4, 2, 1):
        FAMS |= set(fd["test_families"]); fam_fold.update({f: fd["fold"] for f in fd["test_families"]})
rep = {"families": len(FAMS)}
gt = json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))
g421 = [f for f in gt["features"] if site_family(f["properties"]["region_name"]) in FAMS]
gsp = [f for f in g421 if f["properties"].get("markup_type") == "sp_or"]
for name, feats in (("gt_421.geojson", g421), ("gt_421_sp.geojson", gsp)):
    json.dump({"type": "FeatureCollection", "features": feats}, open(name, "w", encoding="utf-8"), ensure_ascii=False)
rep["gt_421"] = len(g421); rep["gt_421_sp"] = len(gsp); rep["gt_421_regions"] = len({f["properties"]["region_name"] for f in g421})
rep["gt_421_by_class"] = dict(collections.Counter(f["properties"]["class_name"] for f in g421))
rep["gt_421_sp_by_class"] = dict(collections.Counter(f["properties"]["class_name"] for f in gsp))
del gt, g421, gsp
def read_feats(path):
    s = open(path, encoding="utf-8").read(); i = s.index('"features": [') + 13; dec = json.JSONDecoder()
    while True:
        while s[i] in " \n\r\t,": i += 1
        if s[i] == "]": return
        o, i = dec.raw_decode(s, i); yield o
cand = collections.defaultdict(list); ell = collections.defaultdict(list); bad_fam = collections.Counter(); fixed = 0
for fold in (4, 2, 1):
    for f in read_feats(SRC / f"oof_dfine_f{fold}.geojson"):
        p = f["properties"]; reg = p["region_name"]; fam = site_family(reg)
        if fam not in FAMS or fam_fold[fam] != fold: bad_fam[f"{fold}|{reg}"] += 1; continue
        g = shape(f["geometry"])
        if not g.is_valid: g = g.buffer(0); fixed += 1
        x0, y0, x1, y1 = g.bounds
        cand[(reg, p["class_name"])].append(((x0 + x1) / 2, (y0 + y1) / 2, float(p["confidence"]), (x0, y0, x1, y1)))
        if not g.is_empty:
            ell[fam].append({"type": "Feature", "geometry": mapping(g), "properties": {"class_name": p["class_name"],
                "confidence": p["confidence"], "region_name": reg, "crs": "EPSG:3857", "markup_type": "sp_or", "threshold_group": "sp_dfine"}})
rep["outside_family_fold"] = dict(bad_fam); rep["invalid_fixed_buffer0"] = fixed
def cover(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0.0
    sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])); return w * h / sm if sm > 0 else 0.0
CELL = 100.0; MERGE = 4.0; R = 4.0
def cells_of(bx, pad):
    return {(i, j) for i in range(int(math.floor((bx[0] - pad) / CELL)), int(math.floor((bx[2] + pad) / CELL)) + 1)
            for j in range(int(math.floor((bx[1] - pad) / CELL)), int(math.floor((bx[3] + pad) / CELL)) + 1)}
circ = collections.defaultdict(list); n_in = n_out = 0; views = collections.Counter()
for (reg, cls), items in sorted(cand.items()):
    items.sort(key=lambda c: -c[2]); grid = collections.defaultdict(list); clusters = []
    for c in items:
        n_in += 1; x, y, sc, bx = c
        hit = None; seen = set()
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
        b = cl[0]; n_out += 1; views[min(len(cl), 5)] += 1
        geo = Point(b[0], b[1]).buffer(R * mercator_scale(b[1]), quad_segs=4)
        circ[site_family(reg)].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[round(u, 2), round(v, 2)] for u, v in geo.exterior.coords]]},
            "properties": {"class_name": cls, "confidence": round(b[2], 6), "region_name": reg, "crs": "EPSG:3857", "markup_type": "sp_or",
                           "threshold_group": "sp_dfine", "views": len(cl)}})
rep["circles_in"] = n_in; rep["circles_out"] = n_out; rep["views_hist"] = dict(views)
rep["circles_by_class"] = dict(collections.Counter(f["properties"]["class_name"] for v in circ.values() for f in v))
for d in ("fam_base", "fam_C", "fam_E", "fam_Conly"): (OUT / d).mkdir(exist_ok=True)
n_base = 0
for fam in sorted(FAMS):
    src = B10 / "work/fam" / f"{fam}.jsonl"
    base_lines = []
    if src.exists():
        dst = OUT / "fam_base" / f"{fam}.jsonl"
        if not dst.exists(): os.symlink(src, dst)
        base_lines = open(src, encoding="utf-8").read().splitlines(); n_base += len(base_lines)
    for d, extra in (("fam_C", circ.get(fam, [])), ("fam_E", ell.get(fam, [])), ("fam_Conly", circ.get(fam, []))):
        with open(OUT / d / f"{fam}.jsonl", "w", encoding="utf-8") as fh:
            if d != "fam_Conly":
                for l in base_lines: fh.write(l + "\n")
            for f in extra: fh.write(json.dumps(f, ensure_ascii=False) + "\n")
rep["base_feats_421"] = n_base; rep["fams_with_base"] = len(list((OUT / "fam_base").glob("*.jsonl")))
rep["maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
json.dump(rep, open("prep_metric.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False))
