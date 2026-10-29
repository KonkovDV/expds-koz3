# Только эталон sp_or семей фолдов 4,2,1: база A (весь стек), база+D-FINE (C), D-FINE отдельно; honest-10 и LOFO, unseen=drop.
import sys, json
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path[:0] = [str(B10 / "repo"), str(B10 / "repo/scripts")]
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
H = Path(__file__).resolve().parent; out = {}
V = {n: load_curves(H / f"curves_{n}_sp") for n in ("base", "C", "Conly")}
regs = set().union(*(regions_of(c) for c in V.values()))
for ln, k in (("k10", 10), ("lofo", None)):
    folds = M.family_folds(regs, site_family, k)
    for n, cv in V.items():
        c, info = M.honest_counts(cv, folds, unseen="drop"); sc, per = M.score_counts(c)
        out[f"{n}_{ln}"] = {"all": round(sc, 6), "f1": {x: round(per[x]["f1"], 4) for x in per if per[x].get("tp", 0) + per[x].get("fn", 0) > 0},
                           "tp_fp_fn": {x: [int(per[x]["tp"]), int(per[x]["fp"]), int(per[x]["fn"])] for x in per if per[x].get("tp", 0) + per[x].get("fn", 0) > 0}}
        print(n, ln, out[f"{n}_{ln}"], flush=True)
json.dump(out, open("sp_compare.json", "w"), ensure_ascii=False, indent=1)
