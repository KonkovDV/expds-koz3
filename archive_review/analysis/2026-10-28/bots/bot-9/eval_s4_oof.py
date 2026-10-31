#!/usr/bin/env python3
"""AUC (rank, со связями) OOF crop-verify s4: всего, по блокам fold и по семьям (site_family = колонка family).
    python3 eval_s4_oof.py /workspace/kaggle_out/koz3-gpu-s4 > s4_eval.json"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path

def auc(y, p):
    n1 = sum(y); n0 = len(y) - n1
    if not n1 or not n0: return None
    o = sorted(range(len(p)), key=lambda i: p[i]); r = [0.0] * len(p); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and p[o[j + 1]] == p[o[i]]: j += 1
        for t in range(i, j + 1): r[o[t]] = (i + j) / 2 + 1
        i = j + 1
    return round((sum(r[k] for k in range(len(y)) if y[k]) - n1 * (n1 + 1) / 2) / (n1 * n0), 4)

base = Path(sys.argv[1]); out = {}
for var in ("r18_scratch", "r18_imnet"):
    for scheme in ("fam10", "lofo"):
        f = next(iter(base.rglob(f"cropverify/{var}/oof_{scheme}.csv")), None)
        if f is None: out[f"{var}/{scheme}"] = "нет файла"; continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        y = [int(r["y"]) for r in rows]; p = [float(r["p"]) for r in rows]
        by = lambda key: {k: {"n": len(v), "pos": sum(y[i] for i in v), "auc": auc([y[i] for i in v], [p[i] for i in v])}
                         for k, v in sorted(groups(key).items())}
        def groups(key):
            g = defaultdict(list)
            for i, r in enumerate(rows): g[r[key]].append(i)
            return g
        fam = by("family"); vals = [v["auc"] for v in fam.values() if v["auc"] is not None]
        # damaged vs intact среди положительных: y_sub 1 = povr бит, 2 = tselye бит (модель этому не училась — справочно)
        sub = [i for i in range(len(rows)) if int(rows[i]["y_sub"]) in (1, 2)]
        out[f"{var}/{scheme}"] = {"file": str(f), "n": len(y), "pos": sum(y), "auc_all": auc(y, p),
            "auc_by_fold": by("fold"), "by_family": fam,
            "family_macro_auc": round(sum(vals) / len(vals), 4) if vals else None, "families_with_auc": len(vals),
            "ref_auc_povr_vs_tsel_among_pos": auc([int(rows[i]["y_sub"]) == 1 for i in sub], [p[i] for i in sub])}
print(json.dumps(out, ensure_ascii=False, indent=1))
