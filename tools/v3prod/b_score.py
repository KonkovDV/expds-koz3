#!/usr/bin/env python3
"""W1-V3PROD процесс B: финальные выходы → официальное жадное сопоставление (global) → счётчики участков → балл, бутстреп семей."""
import importlib.util, json, sys
from pathlib import Path
_s = importlib.util.spec_from_file_location("d3_gate", "/workspace/koz3/orch/bot-W1/work/d3_gate.py")
D = importlib.util.module_from_spec(_s); sys.modules["d3_gate"] = D; _s.loader.exec_module(D)
M = D.M
T = json.load(open("t_taus.json")); regs = set(T["regs"])
SL = {"all": None, "new34": set(T["new"]), "nolidar": set(T["optical"])}
gt = {"type": "FeatureCollection", "features": json.loads(Path("/workspace/koz3/w5data/gt_train_v2.geojson").read_bytes())["features"]}


GTR = {cl: {r: v[2] for r, v in d.items()} for cl, d in T["counts_B0"].items()}   # формат гейта: [tp, fp, n_gt]
assert all(GTR[cl].get(r, 0) == v[2] for cl, d in T["counts_V3P"].items() for r, v in d.items())


def counts(path):
    fc = json.loads(Path(path).read_bytes())
    cur = M.build_curves(fc, gt, D.METRIC, "global")
    out = {}
    for cl, c in cur.items():
        per = {}
        for r, v in GTR.get(cl, {}).items():      # эталон участков — как в гейте (n_gt счётчиков гейта), не только участки с выдачей
            per.setdefault(r, [0, 0, 0])[2] += int(v)
        for r, h in zip(c.pred_region, c.hit):
            lst = per.setdefault(r, [0, 0, 0]); lst[0 if h else 1] += 1
        out[cl] = per
    return out


res = {}
for v in sys.argv[1:]:
    mode = v
    a = counts(f"/workspace/koz3/w1/v3audit/out_v2c_{mode}.geojson")
    b = counts(f"/workspace/koz3/w1/v3audit/out_v3_{mode}.geojson") if Path(f"/workspace/koz3/w1/v3audit/out_v3_{mode}.geojson").exists() else None
    r = {"v2c": {k: M.score_counts(a, s)[0] for k, s in SL.items()}, "v2c_cls": {c: round(x["f1"], 6) for c, x in M.score_counts(a)[1].items()}}
    if b is not None:
        r["v3"] = {k: M.score_counts(b, s)[0] for k, s in SL.items()}
        r["v3_cls"] = {c: round(x["f1"], 6) for c, x in M.score_counts(b)[1].items()}
        for k, s in SL.items():
            bs = M.paired_bootstrap_counts(b, a, D.site_family, 2000, 0, s)
            r.setdefault("delta", {})[k] = {"d": r["v3"][k] - r["v2c"][k], "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    r["gap_vs_gate"] = {"v2c": r["v2c"]["all"] - T["B0"], **({"v3": r["v3"]["all"] - T["V3P"],
                        "delta": r["v3"]["all"] - r["v2c"]["all"] - (T["V3P"] - T["B0"])} if b is not None else {})}
    r["GAP_FLAG"] = any(abs(x) > 0.003 for x in r["gap_vs_gate"].values())
    res[mode] = r
    print(mode, json.dumps(r, ensure_ascii=False), flush=True)
Path("b_score_" + "_".join(sys.argv[1:]) + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
