# Балл D-FINE отдельно: только круги D-FINE, эталон только sp_or семей фолдов 4,2,1; honest-10 / LOFO (пороги без своего фолда) и in-sample.
import sys, json
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path[:0] = [str(B10 / "repo"), str(B10 / "repo/scripts")]
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
cv = load_curves(Path(__file__).resolve().parent / "curves_Conly_sp"); regs = regions_of(cv)
out = {"regions": len(regs)}
for ln, k in (("k10", 10), ("lofo", None)):
    c, info = M.honest_counts(cv, M.family_folds(regs, site_family, k), unseen="drop"); sc, per = M.score_counts(c)
    out[ln] = {"all": sc, "per_class": {k2: {x: per[k2][x] for x in ("f1", "tp", "fp", "fn") if x in per[k2]} for k2 in per}, "info": info}
c, info = M.honest_counts(cv, [set(regs)] + [set()], unseen="drop") if False else (None, None)
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open("standalone.json", "w"), ensure_ascii=False, indent=1)
