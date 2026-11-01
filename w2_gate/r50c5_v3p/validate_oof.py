"""Validate r50c5 OOF: 5 folds, region_fold match, counts. No metric."""
import json, collections, sys
R = "/workspace/koz3/w2/oof/r50c5/"
A = R + "elenakonkova_koz3-w2-cnx-r50c5-f01/"; B = R + "dmitrijkv_koz3-w2-cnx-r50c5-f234/"
rfA = json.load(open(A + "region_fold.json"))["region_fold"]; rfB = json.load(open(B + "region_fold.json"))["region_fold"]
out = {"region_fold_equal": rfA == rfB, "n_regions_map": len(rfA)}
src = {0: A, 1: A, 2: B, 3: B, 4: B}
feats, per = [], {}
bad = []
for k, d in src.items():
    F = json.load(open(f"{d}oof_cnx_c5-sc_f{k}.geojson"))["features"]
    regs = collections.Counter(f["properties"]["region_name"] for f in F)
    wrong = {r: rfA.get(r) for r in regs if rfA.get(r) != k}
    bad += list(wrong.items())
    gt = json.load(open(f"{d}gt_li_cnx_c5-sc_f{k}.geojson"))["features"]
    gregs = {f["properties"]["region_name"] for f in gt}
    exp = {r for r, v in rfA.items() if v == k}
    per[k] = {"n_pred": len(F), "pred_regions": len(regs), "map_regions": len(exp), "gt_regions": len(gregs),
              "pred_regions_not_in_fold": wrong, "map_regions_without_pred": sorted(exp - set(regs)),
              "gt_regions_eq_map": gregs <= exp, "classes": dict(collections.Counter(f["properties"]["class_name"] for f in F))}
    feats += F
# combined files == union of fold files
cA = len(json.load(open(A + "oof_cnx_c5-sc.geojson"))["features"]); cB = len(json.load(open(B + "oof_cnx_c5-sc.geojson"))["features"])
out.update(per_fold=per, combined_f01=cA, combined_f234=cB, sum_f01=per[0]["n_pred"] + per[1]["n_pred"],
           sum_f234=per[2]["n_pred"] + per[3]["n_pred"] + per[4]["n_pred"], total=len(feats), folds_present=sorted(per),
           region_dup=[r for r, n in collections.Counter((f["properties"]["region_name"], rfA[f["properties"]["region_name"]]) for f in feats).items() if False],
           conf_min=min(f["properties"]["confidence"] for f in feats), groups=dict(collections.Counter(f["properties"].get("threshold_group") for f in feats)))
out["OK"] = out["region_fold_equal"] and not bad and out["combined_f01"] == out["sum_f01"] and out["combined_f234"] == out["sum_f234"] and len(per) == 5
json.dump(out, open("oof_validation.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False)[:3000])
