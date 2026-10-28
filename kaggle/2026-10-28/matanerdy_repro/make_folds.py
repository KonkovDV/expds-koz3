"""Фолды для репро MataNerdy: семьи (expds.naming.site_family) + кластеры соседних семей bot-10,
семьи из configs/folds_li_tin_59_t128.json остаются в своих фолдах, прочие — жадно по числу объектов GT.
Плюс проверка утечки сплита конкурента (регион-сплит → общие семьи в train и val)."""
import json, re, sys, collections
from pathlib import Path
B = Path("/workspace/koz3_shared/bot-9"); sys.path.insert(0, str(B / "repo"))
from expds.naming import site_family
MODS = {"Li": "li", "Ae": "ae", "SpOr": "sp_or"}
regs = collections.defaultdict(lambda: collections.Counter())
for line in open(B / "s3/listing.txt", encoding="utf-8"):
    p = line.split()
    if len(p) < 4: continue
    k = p[-1]; r = k.split("/")[0]
    if k.endswith("_Li_g.tif") or (k.lower().endswith(".tif") and ("_Ae_" in k or "_SpOr_" in k)):
        regs[r]["tif_bytes"] += int(p[-2])
gt = json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))["features"]
K5 = {"kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"}
for f in gt:
    q = f["properties"]
    if q["class_name"] in K5 and q["markup_type"] in MODS.values() and q["region_name"] in regs:
        regs[q["region_name"]]["gt"] += 1; regs[q["region_name"]]["gt_" + q["class_name"]] += 1
cl = json.load(open("/workspace/koz3_shared/bot-10/work/family_clusters.json", encoding="utf-8"))["clusters"]
fam2cl = {m: c for c, ms in cl.items() for m in ms}
fam = {r: fam2cl.get(site_family(r), site_family(r)) for r in regs}
old = json.load(open(B / "configs/folds_li_tin_59_t128.json", encoding="utf-8"))["folds"]
f_of = {}
for fo in old:
    for fm in fo["test_families"]: f_of[fam2cl.get(fm, fm)] = fo["fold"]
size = collections.Counter(); famgt = collections.Counter(fam[r] for r in regs for _ in range(regs[r]["gt"]))
for fm, k in f_of.items(): size[k] += famgt[fm]
for fm in sorted(set(fam.values()) - set(f_of), key=lambda x: (-famgt[x], x)):
    k = min(range(5), key=lambda i: (size[i], i)); f_of[fm] = k; size[k] += famgt[fm]
folds = []
for k in range(5):
    test = sorted(r for r in regs if f_of[fam[r]] == k)
    trf = collections.Counter({fm: famgt[fm] for fm in set(fam.values()) if f_of[fm] != k})
    val_family = trf.most_common(1)[0][0]  # самая крупная семья обучения → ранняя остановка и пороги
    folds.append({"fold": k, "test_families": sorted({fam[r] for r in test}), "val_family": val_family,
                  "test_regions": test, "test_gt": sum(regs[r]["gt"] for r in test),
                  "test_tif_GB": round(sum(regs[r]["tif_bytes"] for r in test) / 1e9, 2)})
out = {"source": "make_folds.py (s3/listing.txt, gt_train_v2, bot-10 family_clusters, folds_li_tin_59_t128)",
       "family_of": fam, "folds": folds,
       "regions": {r: dict(regs[r]) for r in sorted(regs)}}
json.dump(out, open("folds_matanerdy_k5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for f in folds: print(f["fold"], len(f["test_regions"]), "регионов", f["test_gt"], "GT", f["test_tif_GB"], "ГБ", "val:", f["val_family"])
print("всего регионов с Li_g/Ae/SpOr:", len(regs), "с GT 5 классов:", sum(1 for r in regs if regs[r]["gt"]), "ГБ:", round(sum(v["tif_bytes"] for v in regs.values())/1e9, 2))
# --- утечка у конкурента ---
import csv
S = Path("/workspace/koz3_shared/competitors/MataNerdy-Geodata_Archaeology_CV/03_multiclass_segmentation_deeplab/splits/archaeology_5class_research_split_v1")
def regions_of(fn): return {row["region"] for row in csv.DictReader(open(S / fn, encoding="utf-8"))}
tr, va = regions_of("train_split.csv"), regions_of("val_split.csv")
fam_c = lambda r: fam2cl.get(site_family(r), site_family(r))
common_reg = tr & va
shared = sorted({fam_c(r) for r in tr} & {fam_c(r) for r in va})
leak = {fm: {"train": sorted(r for r in tr if fam_c(r) == fm), "val": sorted(r for r in va if fam_c(r) == fm)} for fm in shared}
json.dump({"train_regions": len(tr), "val_regions": len(va), "regions_in_both": sorted(common_reg), "families_in_both": leak},
          open("competitor_split_leak.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("конкурент: train", len(tr), "val", len(va), "общих регионов", len(common_reg), "общих семей", len(shared))
for fm, v in leak.items(): print(" ", fm, "train:", v["train"][:4], "val:", v["val"][:4])
