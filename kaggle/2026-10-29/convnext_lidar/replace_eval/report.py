# Честные пороги (k10 по семьям, как work4/paired.py: подбор на 9 фолдах, применение к отложенному = leave-family-out):
# пороги групп li_cnx / li по фолдам, Δ honest по классам, по фолдам CNN и по затронутым семьям, TP/FP курганов.
import sys, os, json
from pathlib import Path
B10 = Path("/workspace/koz3_shared/bot-10"); sys.path.insert(0, str(B10 / "repo")); sys.path.insert(0, str(B10 / "repo/scripts"))
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
OUT = Path(__file__).resolve().parent; WD = Path(os.environ.get("RE_WORK", OUT / "work")); os.chdir(WD)
G = json.load(open("gt.log")); VS = sys.argv[1:]; FAMS = set(G["fams"])
V = {v: load_curves(WD / f"c_{v}") for v in ["A"] + VS}
regions = set().union(*(regions_of(c) for c in V.values()))
folds = M.family_folds(regions, site_family, 10)
fam_fold = {site_family(r): fo for r, fo in G["region_fold"].items()}
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
rep = {"k10_folds": [], "variants": VS}
for i, held in enumerate(folds):
    ent = {"k": i, "families": sorted({site_family(r) for r in held})}
    for v in ["A"] + VS:
        res = M.sweep_groups({c: M.curve_where(x, regions - held) for c, x in V[v].items()})
        ent[v] = {c: {g: round(float(gv["threshold"]), 4) for g, gv in x["groups"].items() if str(g).startswith("li")} for c, x in res.items() if c in KUR}
    rep["k10_folds"].append(ent)
def sc(cnt, regs=None):
    s, per = M.score_counts(cnt, regs); return {"score": s, "per_class": per}
for pol in ("drop", "keep"):
    C = {v: M.honest_counts(cv, folds, unseen=pol)[0] for v, cv in V.items()}
    s = {"all": {}}
    for v in VS:
        a, b = sc(C["A"]), sc(C[v])
        s["all"][v] = {"d10": round(b["score"] - a["score"], 6), "A": round(a["score"], 6), "V": round(b["score"], 6),
                       "per_class": {c: round(b["per_class"][c]["f1"] - a["per_class"][c]["f1"], 4) for c in b["per_class"]}}
        for scope, regs in [("cnx_fams", {r for r in regions if site_family(r) in FAMS}), ("rest", {r for r in regions if site_family(r) not in FAMS})] + [("cnn_f" + str(fo), {r for r in regions if fam_fold.get(site_family(r)) == fo}) for fo in sorted(set(fam_fold.values()))] + [("fam_" + f, {r for r in regions if site_family(r) == f}) for f in sorted(FAMS)]:
            if not regs: continue
            sa, sb = sc(C["A"], regs), sc(C[v], regs)
            kt = {}
            for c in KUR:
                for nm, cc in (("A", C["A"]), ("V", C[v])):
                    t = [cc[c][r] for r in regs if r in cc.get(c, {})]
                    kt[f"{c[8:12]}_{nm}"] = [int(sum(x[0] for x in t)), int(sum(x[1] for x in t))] if t else None
            s.setdefault(scope, {})[v] = {"d": round(sb["score"] - sa["score"], 4), "kur_tp_fp": kt}
    rep[pol] = s
json.dump(rep, open("report.json", "w"), ensure_ascii=False, indent=1)
for pol in ("drop", "keep"):
    print(pol, json.dumps(rep[pol]["all"], ensure_ascii=False))
