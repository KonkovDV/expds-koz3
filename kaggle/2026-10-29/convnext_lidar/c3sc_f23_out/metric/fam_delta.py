# k10 (как paired.py): пороги li_cnx, выбранные для каждого k10-фолда (подбор на остальных 9 фолдах = честно, leave-family-out),
# и вклад затронутых семей/фолдов CNN в Δ (A vs X/Xx, f23 и f0–f3)
import sys, json
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo")); sys.path.insert(0, str(B10 / "repo/scripts"))
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
D = Path(__file__).resolve().parent
V = {"A": load_curves(D / "c_A"), "X": load_curves(D / "c_X"), "Xx": load_curves(D / "c_Xx"), "4_X": load_curves(D / "c4_X"), "4_Xx": load_curves(D / "c4_Xx")}
regions = set().union(*(regions_of(c) for c in V.values()))
folds = M.family_folds(regions, site_family, 10)
rf = json.load(open("/workspace/kaggle_out/koz3-cnx-li-c3sc-f23/region_fold.json"))["region_fold"]
fam_fold = {}
for r, fo in rf.items(): fam_fold[site_family(r)] = fo
rep = {"k10_folds": []}
for i, held in enumerate(folds):
    fams = sorted({site_family(r) for r in held}); cnn = sorted({fam_fold.get(f) for f in fams if fam_fold.get(f) in (0, 1, 2, 3)})
    ent = {"k": i, "families": fams, "cnn_folds_in_held": cnn}
    for v in ("X", "4_X"):
        res = M.sweep_groups({c: M.curve_where(x, regions - held) for c, x in V[v].items()})
        ent[f"li_cnx_thr_{v}"] = {c: round(float(x["groups"]["li_cnx"]["threshold"]), 4) for c, x in res.items() if "li_cnx" in x["groups"]}
    rep["k10_folds"].append(ent)
C = {}
for v, cv in V.items():
    for pol in ("drop", "keep"):
        C[f"{v}_{pol}"], _ = M.honest_counts(cv, folds, unseen=pol)
def sub(fo): return {r for r in regions if fam_fold.get(site_family(r)) == fo}
for pol in ("drop", "keep"):
    for v in ("X", "Xx", "4_X", "4_Xx"):
        out = {}
        for nm, rs in [(f"cnnf{fo}", sub(fo)) for fo in (0, 1, 2, 3)] + [(fam, {r for r in regions if site_family(r) == fam}) for fam in sorted({site_family(r) for r in rf if rf[r] in (2, 3)})]:
            if not rs: continue
            a, pa = M.score_counts(C[f"A_{pol}"], rs); b, pb = M.score_counts(C[f"{v}_{pol}"], rs)
            out[nm] = {"A": round(a, 5), v: round(b, 5), "d": round(b - a, 5), "per": {k: {"A": [pa[k]["tp"], pa[k]["fp"], round(pa[k]["f1"], 4)], v: [pb[k]["tp"], pb[k]["fp"], round(pb[k]["f1"], 4)]} for k in pa if k in pb and k.startswith(("kurg", "gor", "fort"))}}
        rep[f"{v}_{pol}"] = out
json.dump(rep, open("fam_delta.json", "w"), ensure_ascii=False, indent=1)
for e in rep["k10_folds"]: print(e["k"], e["cnn_folds_in_held"], len(e["families"]), e["li_cnx_thr_X"], e["li_cnx_thr_4_X"])
for v in ("X", "Xx", "4_X", "4_Xx"): print(v, {k: x["d"] for k, x in rep[f"{v}_drop"].items()})
