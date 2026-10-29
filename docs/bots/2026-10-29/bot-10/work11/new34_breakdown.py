"""Разбор Δ на new34 (regions_las_new.txt) по классам и участкам: k10 honest, keep. usage: new34_breakdown.py base_dir var_dir out.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
bd, vd, out = sys.argv[1:4]
new = {l.strip() for l in Path("repo/notes/regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
B, V = load_curves(Path(bd)), load_curves(Path(vd))
regions = regions_of(B) | regions_of(V)
folds = M.family_folds(regions, site_family, 10)
cb, _ = M.honest_counts(B, folds, unseen="keep"); cv, _ = M.honest_counts(V, folds, unseen="keep")
sb, pb = M.score_counts(cb, new); sv, pv = M.score_counts(cv, new)
res = {"new34_base": sb, "new34_var": sv, "d": sv - sb, "per_class": {}}
for k in pb:
    res["per_class"][k] = {"base": {x: pb[k].get(x) for x in ("f1", "tp", "fp", "n_gt") if x in pb[k]},
                           "var": {x: pv[k].get(x) for x in ("f1", "tp", "fp", "n_gt") if x in pv[k]}}
# поучастково: TP/FP по классам в new34
per_reg = {}
for r in sorted(new & regions):
    sbr, pbr = M.score_counts(cb, {r}); svr, pvr = M.score_counts(cv, {r})
    diff = {k: (pvr[k]["tp"] - pbr[k]["tp"], pvr[k]["fp"] - pbr[k]["fp"]) for k in pbr if k in pvr and (pvr[k]["tp"] != pbr[k]["tp"] or pvr[k]["fp"] != pbr[k]["fp"])}
    if diff: per_reg[r] = {"score_base": sbr, "score_var": svr, "dTP_dFP": diff}
res["per_region"] = per_reg
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ("new34_base", "new34_var", "d")}), len(per_reg))
