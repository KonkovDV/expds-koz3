# Входы для замера li_cnx (c3-sc, фолды 2,3) на конвейере бота 10 (bot-10 только чтение). Обобщение c3sc_f01_out/metric/prep.py.
# Затронутые семьи = site_family всех участков OOF f2/f3 (+ все члены затронутых кластеров family_clusters.json).
#   gt            -> gt_t.geojson (весь GT затронутых семей), gt_li.geojson (только markup li)
#   fam V         -> fam_V/{семья}.jsonl:  base = симлинки на bot-10/work/fam;  X = база + все cnx (threshold_group li_cnx);
#       Xx = база + cnx с подавлением «один центр — два класса курганов» (больший confidence);  cnx / cnxx = только cnx
import sys, json, os, collections
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo"))
from expds.naming import site_family
OUT = Path(__file__).resolve().parent; os.chdir(OUT)
K = Path("/workspace/kaggle_out/koz3-cnx-li-c3sc-f23"); KUR = {"kurgany_povrezhdennye", "kurgany_tselye"}
oof = json.load(open(K / "oof_cnx_c3-sc.geojson", encoding="utf-8"))["features"]
CL = json.load(open(B10 / "work/family_clusters.json", encoding="utf-8"))["clusters"]
FAMS = {site_family(f["properties"]["region_name"]) for f in oof}
for k, mem in CL.items():
    if FAMS & set(mem): FAMS |= set(mem)
FAMS = sorted(FAMS)
cmd = sys.argv[1]
if cmd == "gt":
    gt = json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))["features"]
    gt_t = [f for f in gt if site_family(f["properties"]["region_name"]) in FAMS]
    gli = [f for f in gt_t if f["properties"].get("markup_type") == "li"]
    for n, fs in (("gt_t.geojson", gt_t), ("gt_li.geojson", gli)):
        json.dump({"type": "FeatureCollection", "features": fs}, open(n, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({"fams": FAMS, "n_fams": len(FAMS), "gt_t": len(gt_t), "gt_li": len(gli), "gt_li_cls": collections.Counter(f["properties"]["class_name"] for f in gli)}, ensure_ascii=False))
    sys.exit()
V = sys.argv[2]; d = OUT / f"fam_{V}"; d.mkdir(exist_ok=True)
if V in ("Xx", "cnxx"):
    best = {}
    for i, f in enumerate(oof):
        p = f["properties"]
        if p["class_name"] not in KUR: continue
        cs = f["geometry"]["coordinates"][0]; key = (p["region_name"], round(sum(c[0] for c in cs) / len(cs), 1), round(sum(c[1] for c in cs) / len(cs), 1))
        if key not in best or oof[best[key]]["properties"]["confidence"] < p["confidence"]: best[key] = i
    keep = set(best.values())
    oof = [f for i, f in enumerate(oof) if f["properties"]["class_name"] not in KUR or i in keep]
extra = collections.defaultdict(list)
for f in oof:
    p = f["properties"]; assert p.get("threshold_group") == "li_cnx"
    extra[site_family(p["region_name"])].append(json.dumps({"type": "Feature", "geometry": f["geometry"], "properties": {k: p[k] for k in ("class_name", "confidence", "region_name", "crs", "markup_type", "threshold_group")}}, ensure_ascii=False))
rep = {"variant": V, "cnx_feats": sum(len(v) for v in extra.values())}
for m in FAMS:
    src = B10 / "work/fam" / f"{m}.jsonl"; dst = d / f"{m}.jsonl"
    if dst.exists() or dst.is_symlink(): dst.unlink()
    if V == "base": os.symlink(src, dst); continue
    nb = 0
    with open(dst, "w", encoding="utf-8") as fh:
        if V in ("X", "Xx"):
            for l in open(src, encoding="utf-8"):
                fh.write(l if l.endswith("\n") else l + "\n"); nb += 1
        for l in extra.get(m, []): fh.write(l + "\n")
    rep[m] = {"base_lines": nb, "cnx": len(extra.get(m, []))}
rep["size_MB"] = round(sum(p.stat().st_size for p in d.glob("*.jsonl") if not p.is_symlink()) / 2**20, 1)
json.dump(rep, open(f"fam_{V}.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False))
