"""Честный балл при unseen=drop и unseen=keep: уровни, парные Δ, бутстреп по семьям, раскладки."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of, random_family_folds, boot, fam_index, stats, weighted_class_delta
from expds import metric as M
from expds.naming import site_family
new = {l.strip() for l in Path("repo/notes/regions_las_new.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
V = {"A": load_curves(Path("work/curves")), "nogor": load_curves(Path("work/curves_nogor"))}
regions = set().union(*(regions_of(c) for c in V.values()))
lay = {"k10": M.family_folds(regions, site_family, 10), "lofo": M.family_folds(regions, site_family, None)}
for s in range(5):
    lay[f"seed{s}"] = random_family_folds(regions, 10, s)
rep = {"layouts": {}}
C = {}
for ln, folds in lay.items():
    row = {}
    for nm, cv in V.items():
        for pol in ("drop", "keep"):
            c, info = M.honest_counts(cv, folds, unseen=pol)
            sc, per = M.score_counts(c)
            row[f"{nm}_{pol}"] = {"all": sc, "new34": M.score_counts(c, new)[0], "object_poly_f1": per["object_poly"]["f1"],
                                  "object_poly_tp_fp": [per["object_poly"]["tp"], per["object_poly"]["fp"]], "info": info}
            if ln == "k10":
                C[f"{nm}_{pol}"] = c
    row["delta_keep_minus_drop_A"] = row["A_keep"]["all"] - row["A_drop"]["all"]
    row["delta_A_nogor_drop"] = row["A_drop"]["all"] - row["nogor_drop"]["all"]
    row["delta_A_nogor_keep"] = row["A_keep"]["all"] - row["nogor_keep"]["all"]
    rep["layouts"][ln] = row
    print(ln, {k: (round(v["all"], 6) if isinstance(v, dict) else round(v, 6)) for k, v in row.items()}, flush=True)
fams = fam_index(list(C.values()))
names = ["A_drop", "A_keep", "nogor_drop", "nogor_keep"]
b = boot([C[n] for n in names], 2000, 0, fams)
rep["boot_keep_minus_drop_A"] = stats(b[1] - b[0])
rep["boot_A_nogor_drop"] = stats(b[0] - b[2])
rep["boot_A_nogor_keep"] = stats(b[1] - b[3])
rep["builtin_keep_minus_drop_A"] = M.paired_bootstrap_counts(C["A_keep"], C["A_drop"], site_family, 1000)
rep["per_class_weighted_keep_minus_drop_A"] = weighted_class_delta(C["A_keep"], C["A_drop"])
# вклад одной семьи Гнёздово
gn = {r for r in regions if site_family(r) == "Гнёздово"}
rep["gnezdovo_regions"] = sorted(gn)
rep["without_gnezdovo"] = {n: M.score_counts(C[n], regions - gn)[0] for n in names}
Path("work3/keep_vs_drop.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
for k in ("boot_keep_minus_drop_A", "boot_A_nogor_drop", "boot_A_nogor_keep", "builtin_keep_minus_drop_A", "without_gnezdovo"):
    print(k, rep[k])
print({c: round(v["weighted"], 6) for c, v in rep["per_class_weighted_keep_minus_drop_A"].items()}, len(gn))
