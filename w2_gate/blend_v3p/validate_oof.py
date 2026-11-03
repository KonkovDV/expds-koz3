"""Validate P-W2-12 blend OOF: 5 folds, fold == region_fold (r50c5 + s2 kernels), families not split, counts. No metric."""
import json, collections, glob
B = "/workspace/koz3/w2/blend/"
R = "/workspace/koz3/w2/oof/r50c5/"
rf = json.load(open(R + "elenakonkova_koz3-w2-cnx-r50c5-f01/region_fold.json"))["region_fold"]
rf_s2 = [json.load(open(p))["region_fold"] for p in glob.glob("/workspace/koz3/w2/oof/s2/*/region_fold.json")]
out = {"n_regions_map": len(rf), "region_fold_eq_s2_kernels": all(x == rf for x in rf_s2) and len(rf_s2) > 0}
ok = out["region_fold_eq_s2_kernels"]
for n in ["blend3", "blend4", "check_s2"]:
    F = json.load(open(B + n + ".geojson"))["features"]
    side = json.load(open(B + n + ".geojson.json"))
    P = [f["properties"] for f in F]
    wrong = sorted({p["region_name"] for p in P if rf.get(p["region_name"]) != p["fold"]})
    fam = collections.defaultdict(set)
    for p in P: fam[p["family"]].add(p["fold"])
    split = {k: sorted(v) for k, v in fam.items() if len(v) > 1}
    regs = {p["region_name"] for p in P}
    r = {"n": len(F), "folds": dict(sorted(collections.Counter(p["fold"] for p in P).items())),
         "classes": dict(collections.Counter(p["class_name"] for p in P)),
         "groups": dict(collections.Counter(p["threshold_group"] for p in P)),
         "regions_with_pred": len(regs), "map_regions_without_pred": sorted(set(rf) - regs),
         "pred_regions_fold_mismatch": wrong, "families_split": split,
         "conf_min": min(p["confidence"] for p in P), "conf_max": max(p["confidence"] for p in P),
         "sidecar_regions": len(side.get("per_region", {}))}
    r["OK"] = not wrong and not split and len(r["folds"]) == 5
    ok &= r["OK"]; out[n] = r
out["OK"] = ok
json.dump(out, open("oof_validation.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("n", "folds", "OK", "families_split", "pred_regions_fold_mismatch", "map_regions_without_pred")}) for k, v in out.items()}, ensure_ascii=False))
