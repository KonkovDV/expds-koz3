"""Lidar operating point on honest-10 OOF (stack A, table.pkl): P/R of li* groups at fold-honest thresholds,
and the same at threshold scale s (all li* kurgan thresholds x s) — descriptive only, no gate.
"P" in li_op.json is TP/(TP+FP); the README recomputes it the metric's way, TP/(TP+0.5*FP), from the same tp/fp."""
import importlib.util, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("d3_gate", "/workspace/koz3/orch/bot-W1/work/d3_gate.py")
D = importlib.util.module_from_spec(_s); sys.modules["d3_gate"] = D; _s.loader.exec_module(D)
M, site_family = D.M, D.site_family
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10)
nA, info, taus = D.honest(cA, folds)
print("A honest", M.score_counts(nA)[0], flush=True)
# GT by markup type per class
gmt = Counter()
for f in stk.gt_all:
    p = f.get("properties", {})
    gmt[(p.get("class_name"), p.get("markup_type"))] += 1
print("GT markup types:", sorted({k[1] for k in gmt}))
groups = sorted({g for c in cA.values() for g in c.pred_group})
print("pred groups:", groups)
every = set().union(*folds)
def counts_by_group(scale_li=1.0, li_classes=("kurgany_povrezhdennye", "kurgany_tselye")):
    out = defaultdict(lambda: [0, 0])  # (grp, cls) -> tp, fp
    for fi, held in enumerate(folds):
        t = taus[fi]
        for cls, c in cA.items():
            conf = c.conf; hit = c.hit
            pr = np.array(c.pred_region); pg = np.array(c.pred_group)
            m = np.isin(pr, list(held))
            for g in set(pg[m].tolist()):
                tau = t.get(g, {}).get(cls)
                if tau is None:
                    continue
                if g.startswith("li") and cls in li_classes:
                    tau = tau * scale_li
                mm = m & (pg == g) & (conf >= tau)
                out[(g, cls)][0] += int(hit[mm].sum()); out[(g, cls)][1] += int((~hit[mm]).sum())
    return out
res = {"A_honest": M.score_counts(nA)[0], "gt_by_type": {f"{k[0]}|{k[1]}": v for k, v in gmt.items()}, "scales": {}}
for s in (0.8, 0.9, 1.0, 1.1):
    cb = counts_by_group(s)
    li = [(k, v) for k, v in cb.items() if k[0].startswith("li")]
    tp = sum(v[0] for k, v in li); fp = sum(v[1] for k, v in li)
    ngt_li = sum(v for k, v in gmt.items() if k[1] in ("li", "Li", "lidar"))
    res["scales"][s] = {"tp": tp, "fp": fp, "P": tp / max(tp + fp, 1), "R_approx": tp / max(ngt_li, 1), "n_gt_li": ngt_li,
                        "by": {f"{k[0]}|{k[1]}": v for k, v in sorted(li)}}
    print(s, res["scales"][s]["P"], res["scales"][s]["R_approx"], tp, fp, ngt_li, flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, ensure_ascii=False, indent=1))
