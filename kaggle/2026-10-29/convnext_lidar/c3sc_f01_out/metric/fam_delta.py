# k10 (как paired.py): вклад семей в Δ и пороги li_cnx, выбранные для фолдов с Гнёздово/Тимерево
import sys, json
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo")); sys.path.insert(0, str(B10 / "repo/scripts"))
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
D = Path(__file__).resolve().parent
V = {v: load_curves(D / f"c_{v}") for v in ("A", "X", "Xx")}
regions = set().union(*(regions_of(c) for c in V.values()))
folds = M.family_folds(regions, site_family, 10)
rep = {}
for fam in ("Гнёздово", "Тимерево"):
    held = next(f for f in folds if any(site_family(r) == fam for r in f))
    res = M.sweep_groups({c: M.curve_where(x, regions - held) for c, x in V["X"].items()})
    rep[f"li_cnx_thr_when_{fam}_held"] = {c: v["groups"]["li_cnx"]["threshold"] for c, v in res.items() if "li_cnx" in v["groups"]}
    rep[f"fold_of_{fam}_families"] = sorted({site_family(r) for r in held})
C = {}
for v, cv in V.items():
    for pol in ("drop", "keep"):
        C[f"{v}_{pol}"], _ = M.honest_counts(cv, folds, unseen=pol)
for pol in ("drop", "keep"):
    for v in ("X", "Xx"):
        out = {}
        for fam in ("Гнёздово", "Тимерево"):
            rs = {r for r in regions if site_family(r) == fam}
            a, pa = M.score_counts(C[f"A_{pol}"], rs); b, pb = M.score_counts(C[f"{v}_{pol}"], rs)
            out[fam] = {"A": a, v: b, "d": b - a, "per": {k: {"A": [pa[k]["tp"], pa[k]["fp"], round(pa[k]["f1"], 4)], v: [pb[k]["tp"], pb[k]["fp"], round(pb[k]["f1"], 4)]} for k in pa if k in pb and k.startswith(("kurg", "gor", "fort"))}}
        rep[f"{v}_{pol}"] = out
json.dump(rep, open("fam_delta.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False, indent=0)[:4000])
