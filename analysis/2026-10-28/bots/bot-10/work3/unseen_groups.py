"""Какие (группа, класс) не получают порога на обучающих фолдах и что с ними при drop/keep."""
import sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "repo"); sys.path.insert(0, "repo/scripts")
from honest_bootstrap import load_curves, regions_of
from expds import metric as M
from expds.naming import site_family
cv = load_curves(Path(sys.argv[1]))
regions = set().union(regions_of(cv))
folds = M.family_folds(regions, site_family, 10)
every = set().union(*folds)
stat = defaultdict(lambda: {"n": 0, "tp": 0, "regions": set(), "folds": set(), "conf_min": 1e9})
allgroups = defaultdict(lambda: {"n": 0, "tp": 0, "regions": set()})
for cls, c in cv.items():
    for g, r, h in zip(c.pred_group, c.pred_region, c.hit):
        a = allgroups[(g, cls)]; a["n"] += 1; a["tp"] += int(h); a["regions"].add(r)
for i, held in enumerate(folds):
    res = M.sweep_groups({cls: M.curve_where(c, every - held) for cls, c in cv.items()})
    have = {(g, cls) for cls, v in res.items() for g in v["groups"]}
    for cls, c in cv.items():
        hc = M.curve_where(c, held)
        for g, r, h, cf in zip(hc.pred_group, hc.pred_region, hc.hit, hc.conf):
            if (g, cls) not in have:
                s = stat[(g, cls)]; s["n"] += 1; s["tp"] += int(h); s["regions"].add(r); s["folds"].add(i)
                s["conf_min"] = min(s["conf_min"], float(cf))
out = []
for (g, cls), s in sorted(stat.items(), key=lambda x: -x[1]["n"]):
    a = allgroups[(g, cls)]
    row = {"group": g, "cls": cls, "unseen_n": s["n"], "unseen_tp": s["tp"], "unseen_regions": sorted(s["regions"]),
           "unseen_families": sorted({site_family(r) for r in s["regions"]}), "folds": sorted(s["folds"]),
           "total_n": a["n"], "total_tp": a["tp"], "total_families": sorted({site_family(r) for r in a["regions"]}),
           "conf_min": s["conf_min"]}
    out.append(row)
    print(g, cls, "unseen", s["n"], "tp", s["tp"], "fam", row["unseen_families"], "| total", a["n"], a["tp"], row["total_families"])
Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
