import csv, json, sys, collections
import numpy as np
from sklearn.metrics import roc_auc_score
W = sys.argv[1] if len(sys.argv) > 1 else "/workspace/koz3/cpu/kurgan/work"
rows = list(csv.DictReader(open(f"{W}/kurgan_subclass_oof.csv", encoding="utf-8")))
pcols = [c for c in rows[0] if c.startswith("p_")]
print("cols", pcols, "n", len(rows))
lab = [r for r in rows if r["y"] not in ("", "-1") and r["p_povr"] not in ("", "nan")]
def auc(rs, col="p_povr"):
    y = [int(r["y"]) for r in rs]; p = [float(r[col]) for r in rs]
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
res = {}
for kind in ("all", "gt", "cand"):
    rs = [r for r in lab if kind == "all" or r["kind"] == kind]
    res[kind] = {c: round(auc(rs, c), 4) for c in pcols}; res[kind]["n"] = len(rs)
print("pooled AUC", json.dumps(res, ensure_ascii=False))
fold = collections.defaultdict(list)
for r in lab: fold[r["fold"]].append(r)
print("per-fold AUC(p_povr):", {k: (round(auc(v), 3), len(v)) for k, v in sorted(fold.items(), key=lambda x: int(x[0]))})
# matched candidates: current class vs classifier class, metric-weighted proxy
cand = [r for r in lab if r["kind"] == "cand"]
wt = {1: 30, 0: 20}
def score(pred):  # weighted correct share over matched candidates
    ok = [wt[int(r["y"])] * (pr == int(r["y"])) for r, pr in zip(cand, pred)]
    return sum(ok) / sum(wt[int(r["y"])] for r in cand)
cur = [1 if r["cand_class"] == "kurgany_povrezhdennye" else 0 for r in cand]
print("matched cand", len(cand), "current acc", round(np.mean([c == int(r["y"]) for c, r in zip(cur, cand)]), 4),
      "w-acc", round(score(cur), 4), "AUC within cand", round(auc(cand), 4))
for thr in (0.3, 0.4, 0.5):
    pr = [1 if float(r["p_povr"]) >= thr else 0 for r in cand]
    print(f"relabel@{thr}: acc", round(np.mean([a == int(r['y']) for a, r in zip(pr, cand)]), 4), "w-acc", round(score(pr), 4),
          "flips", sum(a != c for a, c in zip(pr, cur)))
# twin proxy (K2): twins on cand where classifier disagrees confidently
for hi, lo in ((0.6, 0.2), (0.7, 0.15)):
    tp = fp = 0
    for r, c in zip(cand, cur):
        p = float(r["p_povr"]); y = int(r["y"])
        if c == 0 and p >= hi: tp += y == 1; fp += y == 0
        if c == 1 and p <= lo: tp += y == 0; fp += y == 1
    print(f"twin hi={hi} lo={lo}: twins on matched cand -> would-be TP {tp}, FP {fp}")
unl = [r for r in rows if r["kind"] == "cand" and r["y"] in ("", "-1")]
print("unlabeled cand (FP-side) twins at 0.6/0.2:", sum(1 for r in unl if r['p_povr'] not in ('', 'nan') and (
    (r['cand_class'] == 'kurgany_tselye' and float(r['p_povr']) >= .6) or (r['cand_class'] == 'kurgany_povrezhdennye' and float(r['p_povr']) <= .2))), "of", len(unl))
