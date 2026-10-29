# Только эталон sp_or (1606 объектов, 56 затронутых юнитов, все 5 фолдов) и срез фолдов 0,3: база A (весь стек), база+D-FINE (C),
# D-FINE отдельно (Conly), R = база без sp_arch/sp_focal/sp_focal2/sp_or + D-FINE; honest-10 и LOFO, unseen=drop. Кривые с family_clusters.
import sys, json
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path[:0] = [str(B10 / "repo"), str(B10 / "repo/scripts")]
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
H = Path(__file__).resolve().parent; out = {}
T = json.load(open(H / "touched.json")); u03 = set(T["fold_units"]["0"]) | set(T["fold_units"]["3"])
import os
N = ("base", "C", "Conly", "R")
for n in N:  # срез фолдов 0,3: каталоги симлинков на pkl юнитов
    d = H / f"sp03_{n}"; d.mkdir(exist_ok=True)
    for p in (H / f"curves_{n}_sp").glob("*.pkl"):
        if p.stem in u03 and not (d / p.name).exists(): os.symlink(p, d / p.name)
for sub in ("all5", "f03"):
    VV = {n: load_curves(H / (f"curves_{n}_sp" if sub == "all5" else f"sp03_{n}")) for n in N}
    regs = set().union(*(regions_of(c) for c in VV.values()))
    for ln, k in (("k10", 10), ("lofo", None)):
        folds = M.family_folds(regs, site_family, k)
        for n, cv in VV.items():
            c, info = M.honest_counts(cv, folds, unseen="drop"); sc, per = M.score_counts(c)
            out[f"{sub}_{n}_{ln}"] = {"all": round(sc, 6), "units": len(list((H / (f"curves_{n}_sp" if sub == "all5" else f"sp03_{n}")).glob("*.pkl"))), "f1": {x: round(per[x]["f1"], 4) for x in per if per[x].get("tp", 0) + per[x].get("fn", 0) > 0},
                               "tp_fp_fn": {x: [int(per[x]["tp"]), int(per[x]["fp"]), int(per[x]["fn"])] for x in per if per[x].get("tp", 0) + per[x].get("fn", 0) > 0}}
            print(sub, n, ln, out[f"{sub}_{n}_{ln}"], flush=True)
json.dump(out, open(H / "sp_compare5.json", "w"), ensure_ascii=False, indent=1)
