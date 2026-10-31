"""Сводка симуляции: baseline / ядро / multipole (post.py) / патч с выбором параметра вне отложенной семьи."""
import json, glob, re
from collections import defaultdict

def f1(tp, fp, fn):   # как в compute_metrics.F1Calculator: precision = tp/(tp+0.5fp)
    p = tp / (tp + 0.5 * fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)

out = {}
import sys
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
for path in sorted(glob.glob(f"/workspace/koz3_shared/bot-3/sim_poly_rows{TAG}_s*_n*.json")):
    rows = json.load(open(path)); sc = re.search(r"_(s[\d.]+_n[\d.]+)\.json", path).group(1)
    C = defaultdict(lambda: [0, 0, 0])   # (markup, fold, method)
    for r in rows:
        fold = r["family"] if r["markup"] == "maggr" else r["region"]   # etgr/terfl — только Гнёздово: по участкам
        c = C[(r["markup"], fold, r["method"])]; c[0] += r["tp"]; c[1] += r["fp"]; c[2] += r["fn"]
    res = {}; pooled = defaultdict(lambda: [0, 0, 0])
    for mt in ("maggr", "etgr", "terfl"):
        folds = sorted({k[1] for k in C if k[0] == mt}); meths = sorted({k[2] for k in C if k[0] == mt})
        summ = lambda m, fs: [sum(C[(mt, f, m)][i] for f in fs) for i in range(3)]
        rec = {}
        for m in ("baseline", "core_polylabel", "core_dtpole", "multipole_post"):
            rec[m] = summ(m, folds)
        for kind in ("pole", "poly"):
            cands = [m for m in meths if m.startswith("ws_") and m.endswith(kind)]
            tot = [0, 0, 0]; picks = {}
            for f in folds:
                train = [g for g in folds if g != f]
                best = max(cands, key=lambda m: f1(*summ(m, train)))
                picks[f] = best
                tot = [a + b for a, b in zip(tot, summ(best, [f]))]
            rec[f"patch_ws_{kind}_lofo"] = tot
            rec[f"_picks_{kind}"] = sorted(set(picks.values()))
        res[mt] = {k: (v if k.startswith("_") else {"tp": v[0], "fp": v[1], "fn": v[2], "f1": round(f1(*v), 4)}) for k, v in rec.items()}
        for k, v in rec.items():
            if not k.startswith("_"):
                p = pooled[k]; p[0] += v[0]; p[1] += v[1]; p[2] += v[2]
    res["object_poly_all"] = {k: {"tp": v[0], "fp": v[1], "fn": v[2], "f1": round(f1(*v), 4)} for k, v in pooled.items()}
    out[sc] = res
json.dump(out, open(f"/workspace/koz3_shared/bot-3/sim_poly_summary{TAG}.json", "w"), ensure_ascii=False, indent=1)
for sc, res in out.items():
    print("==", sc)
    for mt, rec in res.items():
        print(" ", mt, "  ".join(f"{k.replace('_lofo','')}={v['f1']}({v['tp']}/{v['fp']}/{v['fn']})" for k, v in rec.items() if not k.startswith("_")),
              rec.get("_picks_pole", ""))
