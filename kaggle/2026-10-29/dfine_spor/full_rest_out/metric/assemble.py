# Каталоги кривых (только симлинки): c5_* = все 100 юнитов bot-10/work/curves, где 56 затронутых заменены; c2_* = юниты фолдов 0,3; c3_* = фолды 4,2,1 (сверка с full).
import json, os, sys, pickle, hashlib
from pathlib import Path
M = Path(__file__).resolve().parent; os.chdir(M); A = Path("/workspace/koz3_shared/bot-10/work/curves")
T = json.load(open("touched.json")); units = set(T["units"]); fu = {int(k): set(v) for k, v in T["fold_units"].items()}
allu = {p.stem for p in A.glob("*.pkl")}
rep = {"A_units": len(allu), "touched": len(units), "touched_not_in_A": sorted(units - allu)}
# сверка: база, пересчитанная здесь, == bot-10/work/curves (побайтно или по содержимому)
import numpy as np
def eq(x, y):
    if set(x) != set(y): return False
    for k in x:
        a, b = x[k], y[k]
        if not (np.array_equal(np.sort(a.conf), np.sort(b.conf)) and int(np.sum(a.hit)) == int(np.sum(b.hit)) and a.gt_regions == b.gt_regions): return False
    return True
same_b = same_obj = 0; diff = []
for u in sorted(units):
    a, b = (A / f"{u}.pkl").read_bytes(), (M / "curves_base_t" / f"{u}.pkl").read_bytes()
    if a == b: same_b += 1; continue
    if eq(pickle.loads(a), pickle.loads(b)): same_obj += 1
    else: diff.append(u)
rep["base_check"] = {"bytes_equal": same_b, "obj_equal": same_obj, "differ": diff}
sets = {"c5": allu, "c2": fu[0] | fu[3], "c3": fu[4] | fu[2] | fu[1]}
for sn, us in sets.items():
    for v in ("A", "C", "E", "R", "R0"):
        d = M / f"{sn}_{v}"; d.mkdir(exist_ok=True)
        for p in d.glob("*.pkl"): p.unlink()
        for u in sorted(us & allu):
            src = A / f"{u}.pkl" if (v == "A" or u not in units) else M / f"curves_{v}_t" / f"{u}.pkl"
            os.symlink(src, d / f"{u}.pkl")
    rep[f"{sn}_units"] = len(us & allu); rep[f"{sn}_touched"] = len(us & units)
json.dump(rep, open("assemble.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(rep, ensure_ascii=False))
