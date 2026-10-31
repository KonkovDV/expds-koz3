"""Диагностика gorc: TP/FP/FN gorodishcha (k10, keep) и семьи с ненулевым Δ. python work8/gor_diag.py A=work/curves noc=work8/curves_gorc_noc ..."""
import sys, json
from pathlib import Path
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
V = {}
for s in sys.argv[1:]:
    n, d = s.split("=", 1); V[n] = load_curves(Path(d))
regions = set().union(*(regions_of(c) for c in V.values()))
folds = M.family_folds(regions, site_family, 10)
out = {}
for n, cv in V.items():
    c, info = M.honest_counts(cv, folds, unseen="keep")
    sc, per = M.score_counts(c)
    g = per.get("gorodishcha", {})
    fam = {}
    for f in sorted({site_family(r) for r in regions}):
        rr = {r for r in regions if site_family(r) == f}
        fam[f] = M.score_counts(c, rr)[0]
    out[n] = {"all": sc, "gor": {k: g.get(k) for k in ("tp", "fp", "fn", "f1")}, "fam": fam}
    print(n, round(sc, 6), out[n]["gor"], flush=True)
b = list(V)[0]
for n in list(V)[1:]:
    d = {f: out[n]["fam"][f] - out[b]["fam"][f] for f in out[b]["fam"] if abs(out[n]["fam"][f] - out[b]["fam"][f]) > 1e-9}
    print(n, "семьи с Δ≠0:", {k: round(v, 5) for k, v in sorted(d.items(), key=lambda x: -abs(x[1]))})
json.dump(out, open("work8/gor_diag.json", "w"), ensure_ascii=False, indent=0)
