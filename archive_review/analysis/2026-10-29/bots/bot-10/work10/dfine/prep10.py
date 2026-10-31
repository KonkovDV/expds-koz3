# 5-фолдовый OOF D-FINE (full: 4,2,1 + full_rest: 0,3) и входы для замера на конвейере бота 10 (bot-10 только чтение).
#   merge          -> ../oof_dfine_5f.geojson (group->threshold_group, buffer(0) невалидных, координаты до 0,01 м, fid заново 0..N-1, fid_src)
#   gt             -> touched.json (затронутые семьи/юниты с учётом family_clusters), gt_t.geojson, gt_t_sp.geojson, проверка групп sp_* в базе
#   fam V          -> fam_V/ только для затронутых юнитов: C = база + круги r=4 м (склейка как в coco_preds_to_stack), E = база + эллипсы,
#                     Conly = только круги, base = симлинки на bot-10/work/fam; незатронутые члены кластеров — симлинки на базу.
import sys, json, re, os, math, collections, resource
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo"))
from expds.naming import site_family
from expds.open_ideas import mercator_scale
from shapely.geometry import shape, Point, mapping
OUT = Path(__file__).resolve().parent; os.chdir(OUT)  # bot-10 work10 копия prep5.py bot-9
MERGED = Path("/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest_out/oof_dfine_5f.geojson")
SRCS = [(Path("/workspace/kaggle_out/koz3-dfine-spor-full/oof_dfine_spor.geojson"), (4, 2, 1)),
        (Path("/workspace/kaggle_out/koz3-dfine-spor-full-rest/oof_dfine_spor.geojson"), (0, 3))]
s = open("/workspace/koz3_shared/bot-9/kaggle/dfine_spor/full_rest/koz3-dfine-spor-full-rest.py", encoding="utf-8").read()
F = json.loads(re.search(r"FOLDS = __import__\('json'\)\.loads\(r'''(.*?)'''\)", s, re.S).group(1))
FAM_FOLD = {f: fd["fold"] for fd in F["folds"] for f in fd["test_families"]}
REG_FOLD = {r: fd["fold"] for fd in F["folds"] for r in fd["test_regions"]}
CL = json.load(open(B10 / "work/family_clusters.json"))["clusters"]
UNIT = {}
for members in CL.values():
    for m in members: UNIT[m] = "+".join(sorted(members))
unit_of = lambda fam: UNIT.get(fam, fam)
def read_feats(path):
    s = open(path, encoding="utf-8").read(); i = s.index('"features": [') + 13; dec = json.JSONDecoder()
    while True:
        while s[i] in " \n\r\t,": i += 1
        if s[i] == "]": return
        o, i = dec.raw_decode(s, i); yield o
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
cmd = sys.argv[1]
if cmd == "merge":
    rep = dict(n_in={}, n_out=0, invalid_fixed=0, became_multi=0, empty_dropped=0, invalid_after=0, fold=collections.Counter(), cls=collections.Counter(),
               regions_by_src={}, fold_mismatch=collections.Counter(), not_in_folds=collections.Counter(), keys=collections.Counter())
    fid = 0; fid_src_seen = set(); dup_src = 0
    with open(MERGED, "w", encoding="utf-8") as fh:
        fh.write('{"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}, "features": [\n')
        for path, folds in SRCS:
            n = 0; regs = set()
            for f in read_feats(path):
                n += 1; p = dict(f["properties"]); reg = p["region_name"]; regs.add(reg)
                if p["fold"] not in folds: rep["fold_mismatch"][f"{path.parent.name}|{p['fold']}"] += 1
                if REG_FOLD.get(reg) is None: rep["not_in_folds"][reg] += 1
                elif REG_FOLD[reg] != p["fold"] or FAM_FOLD.get(site_family(reg), FAM_FOLD.get(unit_of(site_family(reg)).split("+")[0])) is None: rep["fold_mismatch"][f"reg|{reg}"] += 1
                k = (p["fold"], p["fid"])
                if k in fid_src_seen: dup_src += 1
                fid_src_seen.add(k)
                p["threshold_group"] = p.pop("group"); p["fid_src"] = p.pop("fid"); p["fid"] = fid
                c = [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in f["geometry"]["coordinates"]]
                g = shape({"type": "Polygon", "coordinates": c})
                if not g.is_valid:
                    g = g.buffer(0); rep["invalid_fixed"] += 1
                    if g.is_empty: rep["empty_dropped"] += 1; continue
                    if g.geom_type != "Polygon": rep["became_multi"] += 1
                    geom = json.loads(json.dumps(mapping(g)))
                    def rr(o): return [rr(x) for x in o] if isinstance(o, (list, tuple)) and o and isinstance(o[0], (list, tuple)) else [round(o[0], 2), round(o[1], 2)]
                    geom["coordinates"] = rr(geom["coordinates"])
                    if not shape(geom).is_valid: rep["invalid_after"] += 1
                else: geom = {"type": "Polygon", "coordinates": c}
                rep["keys"][tuple(sorted(p))] += 1; rep["fold"][p["fold"]] += 1; rep["cls"][p["class_name"]] += 1
                fh.write(("" if fid == 0 else ",\n") + json.dumps({"type": "Feature", "geometry": geom, "properties": p}, ensure_ascii=False, separators=(",", ":")))
                fid += 1
            rep["n_in"][path.parent.name] = n; rep["regions_by_src"][path.parent.name] = sorted(regs)
        fh.write("\n]}\n")
    a, b = (set(v) for v in rep["regions_by_src"].values())
    rep.update(n_out=fid, dup_fid_src_within_fold=dup_src, region_overlap=sorted(a & b), n_regions=len(a | b), keys={"|".join(k): v for k, v in rep["keys"].items()},
               fold=dict(rep["fold"]), cls=dict(rep["cls"]), size_MB=round(MERGED.stat().st_size / 2**20, 1), maxrss_MB=rss())
    rep["regions_by_src"] = {k: len(v) for k, v in rep["regions_by_src"].items()}
    # повторное чтение: уникальность fid, валидность, число объектов
    ids = set(); inval = 0; n2 = 0
    for f in read_feats(MERGED):
        n2 += 1; ids.add(f["properties"]["fid"]); inval += (not shape(f["geometry"]).is_valid)
    rep["reread"] = dict(n=n2, unique_fid=len(ids), invalid=inval)
    json.dump(rep, open("merge.json", "w"), ensure_ascii=False, indent=1, default=dict); print(json.dumps(rep, ensure_ascii=False, default=dict))
elif cmd == "gt":
    fams = collections.Counter(); fam_fold = {}
    for f in read_feats(MERGED):
        fam = site_family(f["properties"]["region_name"]); fams[fam] += 1; fam_fold[fam] = f["properties"]["fold"]
    units = sorted({unit_of(f) for f in fams})
    members = sorted({m for u in units for m in u.split("+")})
    base_dir = B10 / "work/fam"
    # группы sp_* в базе: в каких семьях встречаются (для R/R0 вне затронутых юнитов кривые обязаны совпасть с базой)
    DROP = {"sp_arch", "sp_focal", "sp_focal2", "sp_or"}
    sp_fams = collections.Counter()
    for pth in base_dir.glob("*.jsonl"):
        for l in open(pth, encoding="utf-8"):
            if "sp_" not in l: continue
            pr = json.loads(l)["properties"]; g = str(pr.get("threshold_group") or pr.get("markup_type") or "")
            if g in DROP: sp_fams[pth.stem] += 1
    gt = json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))["features"]
    gt_t = [f for f in gt if site_family(f["properties"]["region_name"]) in members]
    gsp = [f for f in gt_t if f["properties"].get("markup_type") == "sp_or"]
    sp_all = [f for f in gt if f["properties"].get("markup_type") == "sp_or"]
    for name, feats in (("gt_t.geojson", gt_t), ("gt_t_sp.geojson", gsp)):
        json.dump({"type": "FeatureCollection", "features": feats}, open(name, "w", encoding="utf-8"), ensure_ascii=False)
    fold_units = collections.defaultdict(set)
    for fam, fo in FAM_FOLD.items():
        fold_units[fo].add(unit_of(fam))
    rep = dict(dfine_families=len(fams), dfine_fam_by_fold=dict(collections.Counter(fam_fold.values())), units=units, n_units=len(units), member_families=members,
               base_size_MB_members=round(sum((base_dir / f"{m}.jsonl").stat().st_size for m in members if (base_dir / f"{m}.jsonl").exists()) / 2**20, 1),
               gt_t=len(gt_t), gt_t_sp=len(gsp), gt_sp_total=len(sp_all), gt_sp_outside_touched=len(sp_all) - len(gsp),
               gt_t_sp_by_fold=dict(collections.Counter(FAM_FOLD.get(site_family(f["properties"]["region_name"]), FAM_FOLD.get(unit_of(site_family(f["properties"]["region_name"])).split("+")[0], -1)) for f in gsp)),
               sp_group_families=len(sp_fams), sp_group_families_outside_touched=sorted(f for f in sp_fams if f not in members),
               fold_units={k: sorted(v) for k, v in fold_units.items()}, maxrss_MB=rss())
    json.dump(rep, open("touched.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("units", "member_families", "fold_units")}, ensure_ascii=False))
elif cmd == "fam":
    V = sys.argv[2]; T = json.load(open("touched.json")); members = T["member_families"]; base_dir = B10 / "work/fam"
    d = OUT / f"fam_{V}"; d.mkdir(exist_ok=True)
    extra = collections.defaultdict(list); rep = {"variant": V}
    if V in ("C", "Conly"):
        cand = collections.defaultdict(list)
        for f in read_feats(MERGED):
            p = f["properties"]; x0, y0, x1, y1 = shape(f["geometry"]).bounds
            cand[(p["region_name"], p["class_name"])].append(((x0 + x1) / 2, (y0 + y1) / 2, float(p["confidence"]), (x0, y0, x1, y1)))
        def cover(a, b):
            w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
            if w <= 0 or h <= 0: return 0.0
            sm = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])); return w * h / sm if sm > 0 else 0.0
        CELL = 100.0; MERGE = 4.0; R = 4.0
        def cells_of(bx, pad):
            return {(i, j) for i in range(int(math.floor((bx[0] - pad) / CELL)), int(math.floor((bx[2] + pad) / CELL)) + 1)
                    for j in range(int(math.floor((bx[1] - pad) / CELL)), int(math.floor((bx[3] + pad) / CELL)) + 1)}
        n_in = n_out = 0; views = collections.Counter()
        for (reg, cls), items in sorted(cand.items()):
            items.sort(key=lambda c: -c[2]); grid = collections.defaultdict(list); clusters = []
            for c in items:
                n_in += 1; x, y, sc, bx = c; hit = None; seen = set()
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
                extra[site_family(reg)].append(json.dumps({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[round(u, 2), round(v, 2)] for u, v in geo.exterior.coords]]},
                    "properties": {"class_name": cls, "confidence": round(b[2], 6), "region_name": reg, "crs": "EPSG:3857", "markup_type": "sp_or",
                                   "threshold_group": "sp_dfine", "views": len(cl)}}, ensure_ascii=False))
        del cand
        rep.update(circles_in=n_in, circles_out=n_out, views_hist=dict(views))
    elif V == "E":
        for f in read_feats(MERGED):
            p = f["properties"]
            extra[site_family(p["region_name"])].append(json.dumps({"type": "Feature", "geometry": f["geometry"], "properties": {"class_name": p["class_name"],
                "confidence": p["confidence"], "region_name": p["region_name"], "crs": "EPSG:3857", "markup_type": "sp_or", "threshold_group": "sp_dfine"}}, ensure_ascii=False))
    n_extra = 0; n_real = 0
    for m in members:
        src = base_dir / f"{m}.jsonl"; dst = d / f"{m}.jsonl"
        if dst.exists() or dst.is_symlink(): dst.unlink()
        ex = extra.get(m, [])
        if V == "base" or (not ex and V != "Conly"):
            if src.exists(): os.symlink(src, dst)
            continue
        with open(dst, "w", encoding="utf-8") as fh:
            if V != "Conly" and src.exists():
                with open(src, encoding="utf-8") as fs:
                    for l in fs: fh.write(l if l.endswith("\n") else l + "\n")
            for l in ex: fh.write(l + "\n")
        n_extra += len(ex); n_real += 1
    rep.update(extra_feats=n_extra, real_files=n_real, size_MB=round(sum(p.stat().st_size for p in d.glob("*.jsonl") if not p.is_symlink()) / 2**20, 1), maxrss_MB=rss())
    json.dump(rep, open(f"fam_{V}.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False))
