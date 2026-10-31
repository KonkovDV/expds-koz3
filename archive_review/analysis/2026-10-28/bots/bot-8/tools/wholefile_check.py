"""Весь файл сразу: быстрый build_curves/sweep/honest vs официальный calculate_f1 на оставленных; сверка с stack_scoreboard."""
import json, sys, time
sys.path.insert(0, "repo")
from expds import metric as M
from expds.naming import site_family
import importlib.util
spec = importlib.util.spec_from_file_location("ss", "repo/scripts/stack_scoreboard.py"); ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
stack, gtp, proof, out = sys.argv[1:5]
P = json.loads(open(proof).read())
pred = json.loads(open(stack, "rb").read()); gt = json.loads(open(gtp, "rb").read())
mp = "repo/materials/compute_metrics.py"
mod = M.load_official(mp); calc = mod.F1Calculator()
t = time.perf_counter(); curves = M.build_curves(pred, gt, mp); tc = time.perf_counter() - t
taus = ss.sweep_to_taus(M.sweep_groups(curves))
regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
counts, _ = M.honest_counts(curves, M.family_folds(regions, site_family, 10))
honest = M.score_counts(counts)[0]
kept = M.apply_group_thresholds(pred, taus, mp)
fast_all, per = M.score_at(M.build_curves(kept, gt, mp), {})
t = time.perf_counter(); off = calc.calculate_f1(kept, gt); to = time.perf_counter() - t
res = {"n_pred": len(pred["features"]), "n_kept": len(kept["features"]),
       "taus_equal_to_cluster_run": taus == P["taus"], "honest_whole": honest, "honest_cluster": P["honest"][0],
       "fast_all": fast_all, "official_final_score": off["final_score"], "cluster_base_all": P["base"][0],
       "official_counts_equal": all((off["per_class"][c]["tp"], off["per_class"][c]["fp"], off["per_class"][c]["fn"]) == (per[c]["tp"], per[c]["fp"], per[c]["fn"]) for c in calc.target_classes),
       "t_curves_fast": round(tc, 2), "t_official": round(to, 2)}
res["all_equal"] = (res["taus_equal_to_cluster_run"] and honest == P["honest"][0] and fast_all == off["final_score"] == P["base"][0] and res["official_counts_equal"])
print(json.dumps(res, ensure_ascii=False)); open(out, "w").write(json.dumps(res, ensure_ascii=False, indent=1))
