"""SATREC2 modality rule: regions with no lidar (markup li) candidate entering finish (pre- and post-floor, base fold cfg)."""
import copy, json, pickle, sys, tomllib
from pathlib import Path
import numpy as np
X = Path("/workspace/koz3/w1/satrec/x_v3on"); sys.path.insert(0, str(X))
from expds import pipeline as PL
cfg0 = tomllib.load(open(X / "configs/v3_on.toml", "rb"))
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
folds = [set(f) for f in T["folds"]]; regs = set(T["regs"]); dreg = set(T["dreg"])
tab = pickle.load(open("/workspace/koz3/w5data/work/table.pkl", "rb"))
CL = list(tab["CL"]); cls = np.asarray(tab["cls"]); conf = np.asarray(tab["conf"], float); reg = np.asarray(tab["reg"]).astype(str)
grp = np.asarray(tab["grp"]).astype(str); mk = np.asarray(tab["mk"]).astype(str)
FLOORKEY = {"li": "li", "li_mosaic": "li", "li_coarse": "li_coarse", "li_mask": "li_mask", "li_fort": "li_fort", "li_dev": "li_dev"}
KEY = {"ae_fort": "ae", "sp_focal": "sp_or", "sp_arch": "sp_or"}
print("mk values", sorted(set(mk.tolist())), "li grps", sorted(set(grp[mk == "li"].tolist())))
isli = (mk == "li") | np.char.startswith(grp.astype(str), "li")
pre = {r for r in regs if not isli[reg == r].any()}
post = set()
for fi, f in enumerate(folds):
    c = copy.deepcopy(cfg0); bm = c.setdefault("thresholds_by_markup", {})
    for g, d in T["taus_V3P"][fi].items():
        for cl_, v in d.items():
            bm.setdefault(KEY.get(g, g), {})[cl_] = 1.01 if v is None else float(v)
    fl = {k: PL.output_floors(c, k) for k in set(FLOORKEY.values())}
    for r in f:
        idx = np.flatnonzero(isli & (reg == r))
        ok = [conf[i] >= fl[FLOORKEY.get(grp[i], "li")][CL[cls[i]]] for i in idx]
        if not any(ok):
            post.add(r)
opt = set(T["optical"]); new = set(T["new"])
print("NL pre", len(pre), "NL post", len(post), "pre==post", pre == post, "pre==optical", pre == opt, "post-opt", len(post - opt), "opt-post", len(opt - post))
print("dreg: NL", len(dreg & post), "lidar", len(dreg - post), "of which new34", len((dreg - post) & new), "non-new", len((dreg - post) - new))
json.dump({"nl_pre": sorted(pre), "nl_post": sorted(post)}, open("nl_regions.json", "w"), ensure_ascii=False)
