"""Analyse gap_cands.pkl -> runs/gap_breakdown.{json,md}. Diagnostic OOF only."""
import pickle, json, sys, numpy as np
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "main"; sys.path.insert(0, str(ROOT))
from expds import metric as M
from expds.naming import site_family
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
D = pickle.load(open(sys.argv[1], "rb")); C, li, new = D["C"], D["li"], D["new"]; OUT = sys.argv[2]
regions = set()
for A in C.values(): regions |= set(A["reg"]) | set(A["g_reg"])
folds = M.family_folds(regions, site_family, 10); fold_of = {r: k for k, f in enumerate(folds) for r in f}; every = set().union(*folds)
modal = lambda r: "34new" if r in new else ("lidar" if r in li else "optical")
W = M.CLASS_WEIGHTS
def kept_mask(cls, A):
    conf = A["conf"]; reg = A["reg"]; grp = A["grp"]; n = len(conf)
    gnames = sorted(set(grp)); gm = {g: i for i, g in enumerate(gnames)}; gc = np.array([gm[g] for g in grp], np.int64)
    rf = np.array([fold_of.get(r, -1) for r in reg]); keep = np.zeros(n, bool); taus = {}
    gtr = defaultdict(int)
    for r in A["g_reg"]: gtr[r] += 1
    for k, held in enumerate(folds):
        tr = (rf != k) & (rf >= 0); ngt = sum(v for r, v in gtr.items() if r in every - held)
        res = M._sweep_one(conf[tr], A["hit"][tr], gc[tr], gnames, ngt, 20, "max")
        t = {g: gv["threshold"] for g, gv in (res or {}).get("groups", {}).items()}
        if cls == "object_poly": t.update({g: 0.0 for g in gnames if "etgr" in g})
        taus[k] = t; tv = np.array([t.get(g, np.inf) for g in gnames]); m = rf == k
        keep[m] = conf[m] >= tv[gc[m]]
    return keep, taus
cells = defaultdict(lambda: defaultdict(int)); base = {}
KEEP = {cls: kept_mask(cls, A)[0] for cls, A in C.items()}
FIELDS = ["fn_wrong_kept", "fn_wrong_low", "tp", "fp", "ngt", "fp_pure", "fp_nonpoly", "fp_dup_x", "fp_dup_same", "fp_geom", "fn_thr", "fn_assign", "fn_geom", "fn_nocand", "or_hit", "or_bip"]
for cls, A in C.items():
    keep = KEEP[cls]; hit = A["hit"]; asg = A["asg"]; gby = A["g_by"]; grp = A["grp"]
    for i in np.flatnonzero(keep):
        r = A["reg"][i]; c = cells[(cls, r)]
        if hit[i]: c["tp"] += 1; continue
        c["fp"] += 1
        if not A["poly"][i]: c["fp_nonpoly"] += 1; continue
        dj = [j for j in A["okg"][i] if gby[j] >= 0]
        if dj:
            c["fp_dup_x" if any(grp[gby[j]] != grp[i] for j in dj) else "fp_dup_same"] += 1
        elif A["nearg"][i]: c["fp_geom"] += 1
        else: c["fp_pure"] += 1
    rows, cols = [], []
    for j, r in enumerate(A["g_reg"]):
        c = cells[(cls, r)]; c["ngt"] += 1
        b = gby[j]
        if b >= 0: c["or_hit"] += 1
        if b >= 0 and keep[b]: continue
        oks = A["g_ok"][j]
        if b >= 0: c["fn_thr"] += 1
        elif oks: c["fn_assign" if any(keep[i] for i in oks) else "fn_thr"] += 1
        elif A["g_x"][j]: c["fn_wrong_kept" if any(KEEP[c2][i] for c2, i in A["g_x"][j]) else "fn_wrong_low"] += 1
        elif A["g_near"][j]: c["fn_geom"] += 1
        else: c["fn_nocand"] += 1
        for i in oks: rows.append(j); cols.append(i)
    for j in range(len(A["g_reg"])):
        if gby[j] >= 0: rows.append(j); cols.append(int(gby[j]))
    if rows:
        mtx = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(A["g_reg"]), len(A["conf"])))
        mt = maximum_bipartite_matching(mtx, perm_type="column")
        for j in np.flatnonzero(mt >= 0): cells[(cls, A["g_reg"][j])]["or_bip"] += 1
def counts(mod=None, only=None):
    out = defaultdict(dict)
    for (cls, r), c in cells.items():
        tp, fp, g = c["tp"], c["fp"], c["ngt"]
        if mod and (only is None or only(cls, r)): tp, fp = mod(c, tp, fp)
        out[cls][r] = [tp, fp, g]
    return out
sc = lambda cn, regs=None: M.score_counts(cn, regs)[0]
L = {"a_fn_nocand": lambda c, tp, fp: (tp + c["fn_nocand"], fp),
     "c_fn_wrong_class": lambda c, tp, fp: (tp + c["fn_wrong_kept"] + c["fn_wrong_low"], fp),
     "fp_pure": lambda c, tp, fp: (tp, fp - c["fp_pure"] - c["fp_nonpoly"]),
     "b1_fn_threshold": lambda c, tp, fp: (tp + c["fn_thr"], fp),
     "b2_fn_assign_dedup": lambda c, tp, fp: (tp + c["fn_assign"], fp),
     "c_dup_crossmod": lambda c, tp, fp: (tp, fp - c["fp_dup_x"]),
     "c_dup_samemod": lambda c, tp, fp: (tp, fp - c["fp_dup_same"]),
     "d_geometry": lambda c, tp, fp: (tp + c["fn_geom"], fp - c["fp_geom"])}
B0 = sc(counts()); res = {"definition": {"stack": "A+M_rp+ORG3(C19 N0; etgr object_poly tau=0 only), split_arch, honest-10 per-group thresholds, global greedy match (official rule: cov=inter/min area>=0.5 & pred representative point in GT)",
  "oracle_hit": "keep exactly the candidates that the official greedy (full pool, conf order) matches, FP=0", "oracle_bip": "max bipartite matching over ok pairs of all own candidates, FP=0",
  "fp_dup": "kept non-hit candidate with ok pair to a GT already matched; _x = matched by other threshold group (modality)", "fp_geom": "kept non-hit, no ok pair, overlaps a GT (cov>0 / point within R) but fails cov>=0.5 or interior point",
  "fn_thr": "GT matchable by a candidate below honest threshold", "fn_assign": "GT with ok candidate kept but greedy one-to-one gave it to another GT (no pipeline-dedup info in stack)", "fn_geom": "GT with only near candidates (fails rule)", "fn_nocand": "no candidate near", "fn_wrong": "no own-class ok candidate; another class has an ok-pair candidate (_kept: that candidate passes its honest threshold). Precedence: thr > assign > wrong > geom > nocand. Lever adds TP only (FP of the wrong copy not removed)", "dedup": "pipeline dedup happens before the stack, not visible; fn_assign = one-to-one greedy conflict"}}
allr = None; new_r = {r for (_, r) in cells if r in new}; opt_r = {r for (_, r) in cells if modal(r) == "optical"}; lid_r = {r for (_, r) in cells if modal(r) == "lidar"}
res["score"] = {"stack": B0, "stack_34new": sc(counts(), new_r), "stack_optical": sc(counts(), opt_r)}
orc = lambda key: (lambda c, tp, fp: (c[key], 0))
res["score"]["oracle_hit"] = sc(counts(orc("or_hit"))); res["score"]["oracle_bip"] = sc(counts(orc("or_bip")))
res["levers_global"] = {k: round(sc(counts(f)) - B0, 6) for k, f in L.items()}
for nm, regs in (("34new", new_r), ("optical", opt_r), ("lidar_old", lid_r)):
    res[f"levers_{nm}"] = {k: round(sc(counts(f, lambda cl, r, R=regs: r in R)) - B0, 6) for k, f in L.items()}
cls_l = sorted({c for c, _ in cells}); res["levers_by_class"] = {cl: {k: round(sc(counts(f, lambda c_, r, cl=cl: c_ == cl)) - B0, 6) for k, f in L.items()} for cl in cls_l}
res["levers_by_class_modality"] = {f"{cl}|{md}": {k: round(sc(counts(f, lambda c_, r, cl=cl, md=md: c_ == cl and modal(r) == md)) - B0, 6) for k, f in L.items()} for cl in cls_l for md in ("lidar", "34new", "optical")}
tab = []
for (cl, r), c in sorted(cells.items()):
    d = {k: round(sc(counts(f, lambda c_, r_, cl=cl, r=r: c_ == cl and r_ == r)) - B0, 6) for k, f in L.items()}
    tab.append({"class": cl, "w": W.get(cl), "modality": modal(r), "site": r, **{f: c[f] for f in FIELDS}, **{"d_" + k: v for k, v in d.items()}})
res["cells"] = tab; json.dump(res, open(OUT + ".json", "w"), ensure_ascii=False, indent=1)
md = [f"# Gap breakdown (OOF, diagnostic). stack {B0:.6f}; oracle_hit {res['score']['oracle_hit']:.6f}; oracle_bip {res['score']['oracle_bip']:.6f}", "", "Definitions: see json 'definition'. Δ = weighted F1 change when only that loss is removed (others fixed); not additive.", "",
      "| lever | global | lidar_old | 34new | optical |", "|---|---|---|---|---|"]
for k in L: md.append(f"| {k} | {res['levers_global'][k]:+.4f} | {res['levers_lidar_old'][k]:+.4f} | {res['levers_34new'][k]:+.4f} | {res['levers_optical'][k]:+.4f} |")
md += ["", "## class × modality (Δ per lever)", "| class (w) | mod | " + " | ".join(L) + " |", "|---|---|" + "---|" * len(L)]
for key, v in res["levers_by_class_modality"].items():
    cl, mdl = key.split("|"); md.append(f"| {cl} ({W.get(cl)}) | {mdl} | " + " | ".join(f"{v[k]:+.4f}" for k in L) + " |")
agg = defaultdict(lambda: defaultdict(int))
for t in tab:
    for f in FIELDS: agg[(t["class"], t["modality"])][f] += t[f]
md += ["", "## counts class × modality", "| class | mod | " + " | ".join(FIELDS) + " |", "|---|---|" + "---|" * len(FIELDS)]
for (cl, mdl), v in sorted(agg.items()): md.append(f"| {cl} | {mdl} | " + " | ".join(str(v[f]) for f in FIELDS) + " |")
md += ["", "## top-25 site cells by max lever Δ", "| class | mod | site | tp | fp | ngt | best lever | Δ |", "|---|---|---|---|---|---|---|---|"]
for t in sorted(tab, key=lambda t: -max(t["d_" + k] for k in L))[:25]:
    bk = max(L, key=lambda k: t["d_" + k]); md.append(f"| {t['class']} | {t['modality']} | {t['site']} | {t['tp']} | {t['fp']} | {t['ngt']} | {bk} | {t['d_'+bk]:+.4f} |")
open(OUT + ".md", "w").write("\n".join(md) + "\n"); print(json.dumps({k: res[k] for k in ("score", "levers_global", "levers_34new", "levers_optical")}, indent=0)); print(json.dumps(res["levers_by_class"]))
