# li-only по фолду: compute_metrics koz3-1.5.0 (bot-10/repo/materials) на OOF фолда vs gt_li_cnx фолда; копия score_thr из src/convnext_main.py
# (сетка 0.05…0.95 по классу, in-sample по GT теста фолда — оптимистично) + балл без порога. python score_thr.py <fold>
import sys, json, time, logging
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo/materials"); import compute_metrics as CM
logging.getLogger("compute_metrics").setLevel(logging.ERROR)
K = "/workspace/kaggle_out/koz3-cnx-li-c3sc-f23/"; fo = sys.argv[1]
pred = json.load(open(K + f"oof_cnx_c3-sc_f{fo}.geojson", encoding="utf-8")); gt = json.load(open(K + f"gt_li_cnx_c3-sc_f{fo}.geojson", encoding="utf-8"))
GRID = [round(0.05 * k, 2) for k in range(1, 20)]
feats = pred["features"]; cls_present = sorted({f["properties"]["class_name"] for f in feats})
def at(thr):
    keep = [f for f in feats if float(f["properties"]["confidence"]) >= thr.get(f["properties"]["class_name"], 1e9)]
    return CM.F1Calculator().calculate_f1(dict(pred, features=keep), gt)
t0 = time.time(); r0 = CM.F1Calculator().calculate_f1(pred, gt)
best = {}; r03 = None
for t in GRID:
    r = at({c: t for c in cls_present})
    for c, v in r.get("per_class", {}).items():
        if c not in cls_present or not v.get("num_ground_truth"): continue
        if c not in best or v["f1"] > best[c]["f1"]:
            best[c] = dict(thr=t, f1=v["f1"], tp=v["tp"], fp=v["fp"], fn=v["fn"], num_predictions=v["num_predictions"], num_ground_truth=v["num_ground_truth"])
    if t == 0.3: r03 = r
    print(t, round(r["final_score"], 4), round(time.time() - t0), flush=True)
rb = at({c: best[c]["thr"] for c in best})
pc = lambda r: {c: {k: v[k] for k in ("f1", "tp", "fp", "fn", "num_predictions", "num_ground_truth")} for c, v in r["per_class"].items() if v.get("num_predictions") or v.get("num_ground_truth")}
out = dict(fold=int(fo), n_det=len(feats), n_gt=len(gt["features"]), metric=CM.F1Calculator().METRIC_VERSION if hasattr(CM.F1Calculator(), "METRIC_VERSION") else None,
           final_score_nothr=r0["final_score"], per_class_nothr=pc(r0), final_score_thr0_3=r03["final_score"], per_class_thr0_3=pc(r03),
           best_per_class=best, final_score_best_thr=rb["final_score"], per_class_best=pc(rb), t_s=round(time.time() - t0, 1),
           note="in-sample пороги по GT теста фолда — оптимистично")
json.dump(out, open(f"metric/score_thr_f{fo}.json", "w"), ensure_ascii=False, indent=1)
print("f%s n_det %d nothr %.4f thr0.3 %.4f best %.4f" % (fo, len(feats), r0["final_score"], r03["final_score"], rb["final_score"]), {c: (v["thr"], round(v["f1"], 4)) for c, v in best.items()})
