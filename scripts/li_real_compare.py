"""Сверка legacy-декодирования из npz с geojson дампа (задача 9): по (участок, класс) — число объектов,
доля объектов дампа с парой (центроид ближе 0,5 м, тот же класс), |Δconf| у пар, отношение площадей."""
import json
import sys
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

dump, mine, out = sys.argv[1:4]
D = np.load(dump)
M = np.load(mine)
Dreg = np.array(D["regions"])[D["region"]]
Dcls = np.array(D["classes"])[D["cls"]]
LI = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
Mreg = M["region"]
Mcls = np.array(LI)[M["cls"]]
rows, tot = [], Counter()
for r in sorted(set(Dreg) | set(Mreg)):
    for c in LI:
        a = (Dreg == r) & (Dcls == c)
        b = (Mreg == r) & (Mcls == c)
        na, nb = int(a.sum()), int(b.sum())
        if na == 0 and nb == 0:
            continue
        row = {"region": r, "cls": c, "n_dump": na, "n_mine": nb}
        if na and nb:
            t = cKDTree(np.column_stack([M["x"][b], M["y"][b]]))
            d, j = t.query(np.column_stack([D["x"][a], D["y"][a]]))
            ok = d < 0.5
            dc = np.abs(D["conf"][a][ok] - M["conf"][b][j[ok]])
            ar = M["area"][b][j[ok]] / np.maximum(D["area"][a][ok], 1e-9)
            row.update(matched=int(ok.sum()), conf_absdiff_max=float(dc.max()) if dc.size else None,
                       conf_absdiff_mean=float(dc.mean()) if dc.size else None,
                       conf_diff_gt_1e3=int((dc > 1e-3).sum()),
                       area_ratio_p1_p99=[float(np.percentile(ar, 1)), float(np.percentile(ar, 99))] if ar.size else None)
        else:
            row["matched"] = 0
        tot["n_dump"] += na; tot["n_mine"] += nb; tot["matched"] += row["matched"]
        tot["conf_diff_gt_1e3"] += row.get("conf_diff_gt_1e3", 0)
        rows.append(row)
bad = [r for r in rows if r["n_dump"] != r["n_mine"] or r["matched"] != r["n_dump"] or r.get("conf_diff_gt_1e3", 0)]
json.dump({"total": dict(tot), "n_pairs": len(rows), "mismatch_rows": bad, "rows": rows}, open(out, "w"),
          ensure_ascii=False, indent=1)
print(json.dumps(dict(tot), ensure_ascii=False), "mismatch (region,class):", len(bad))
for r in bad[:30]:
    print(r)
