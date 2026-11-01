"""Оценка lidar-aug (c7-sc, elenakonkova/koz3-lidar-aug-f0 v5, фолды 3,4) тем же способом, что f2 и early-stop:
Li-only F1 (compute_metrics koz3-1.5.0) при глобальном пороге confidence>=0.3 по GT теста фолда; против c3-sc s2 (и s3 для f4)
на тех же участках. Запуск: nice -n 10 python3 eval_lidar_aug_f34.py > lidar_aug_f34.json"""
import json, sys, logging, collections
sys.path.insert(0, "/workspace/koz3/w2/vendor")
logging.getLogger("compute_metrics").setLevel(logging.ERROR)
import compute_metrics as CM
O = "/workspace/koz3/out/"
RUNS = {
  "aug_f3": (O+"lidar-aug-f0_v5/oof_cnx_c7-sc_f3.geojson", O+"lidar-aug-f0_v5/gt_li_cnx_c7-sc_f3.geojson"),
  "aug_f4": (O+"lidar-aug-f0_v5/oof_cnx_c7-sc_f4.geojson", O+"lidar-aug-f0_v5/gt_li_cnx_c7-sc_f4.geojson"),
  "c3sc_s2_f3": (O+"koz3-w2-cnx-c3sc-s2f23/oof_cnx_c3-sc_f3.geojson", O+"koz3-w2-cnx-c3sc-s2f23/gt_li_cnx_c3-sc_f3.geojson"),
  "c3sc_s2_f4": (O+"w2-cnx-c3sc-s2f4/oof_cnx_c3-sc_f4.geojson", O+"w2-cnx-c3sc-s2f4/gt_li_cnx_c3-sc_f4.geojson"),
  "c3sc_s3_f4": (O+"w2-cnx-c3sc-s3f4/oof_cnx_c3-sc_f4.geojson", O+"w2-cnx-c3sc-s3f4/gt_li_cnx_c3-sc_f4.geojson"),
}
L = lambda p: json.load(open(p, encoding="utf-8"))
rf = L(O+"lidar-aug-f0_v5/region_fold.json")["region_fold"]
def key(f): return (f["properties"]["region_name"], f["properties"]["class_name"], json.dumps(f["geometry"], sort_keys=True))
out = {"method": "Li-only F1, compute_metrics F1Calculator, global thr conf>=0.3, GT теста фолда (как f2: 0.343 vs c3-sc s2 0.4006)"}
gts = {}
for name, (pp, gp) in RUNS.items():
    pred, gt = L(pp), L(gp)
    regs = sorted({f["properties"]["region_name"] for f in pred["features"]})
    gregs = sorted({f["properties"]["region_name"] for f in gt["features"]})
    gts[name] = (set(map(key, gt["features"])), set(gregs), set(regs))
    keep = [f for f in pred["features"] if float(f["properties"]["confidence"]) >= 0.3]
    r = CM.F1Calculator().calculate_f1(dict(pred, features=keep), gt)
    out[name] = dict(n_pred=len(pred["features"]), n_pred_thr=len(keep), n_gt=len(gt["features"]), n_regions_pred=len(regs),
                     n_regions_gt=len(gregs), folds=dict(collections.Counter(rf.get(x) for x in regs)),
                     score_thr0_3=r.get("final_score"),
                     per_class={c: round(v["f1"], 4) for c, v in r.get("per_class", {}).items() if v.get("num_ground_truth")})
    print(name, out[name]["score_thr0_3"], file=sys.stderr)
chk = {}
for f, ref in (("f3", ["c3sc_s2_f3"]), ("f4", ["c3sc_s2_f4", "c3sc_s3_f4"])):
    a = gts["aug_" + f]
    for r in ref:
        b = gts[r]
        chk[f"aug_{f}_vs_{r}"] = dict(gt_identical=a[0] == b[0], gt_regions_identical=a[1] == b[1], pred_regions_identical=a[2] == b[2])
out["validation"] = chk
print(json.dumps(out, ensure_ascii=False, indent=1))
