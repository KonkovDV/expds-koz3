#!/usr/bin/env python3
"""W4 GA/FP1/GAFP production-replay score (PREREG d366a45) (from W4 q6_score.py <- W1 s3_score.py gate part; PREREG e6b5a31: C1 primary P>=0.9, P2 comparison).
argv: variants (first = base). Reads /workspace/koz3/v4build/gate/out/out_<v>.geojson."""
import importlib.util, json, math, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

ARGV = sys.argv[1:]; sys.argv = sys.argv[:1]
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/v4build/gate/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D = G.D; M = D.M; S = D.S
OUT = Path("/workspace/koz3/orch/bot-W4/comp/satdet/outc1")
T = json.load(open("/workspace/koz3/v4build/res/orch/bot-W1/work/v3prod/t_taus.json"))
folds = [set(f) for f in T["folds"]]; every = set().union(*folds); dreg = set(T["dreg"])
fold_of = {r: i for i, f in enumerate(folds) for r in f}
SL = {"all": None, "new34": set(T["new"]), "nolidar": set(T["optical"])}
gt = {"type": "FeatureCollection", "features": json.loads(Path("/workspace/koz3/w5data/gt_train_v2.geojson").read_bytes())["features"]}
GTR = {cl: {r: v[2] for r, v in d.items()} for cl, d in T["counts_B0"].items()}
TIGHT = {"kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"}
BONF = {'ID': 0.95, 'C1D1f': 0.95, 'C1D2f': 0.95}   # W4 extra vs-C1 comparison (m=2, as dfine2a sidecar); NOT the PREREG family
SP = ("sp_or", "sp_focal", "sp_arch", "sp_focal2")
variants = ARGV or ["base", "C1", "GA", "FP1"]


def counts(fc):
    cur = M.build_curves(fc, gt, D.METRIC, "global")
    out = {}
    for cl, c in cur.items():
        per = {}
        for r, v in GTR.get(cl, {}).items():
            per.setdefault(r, [0, 0, 0])[2] += int(v)
        for r, h in zip(c.pred_region, c.hit):
            lst = per.setdefault(r, [0, 0, 0]); lst[0 if h else 1] += 1
        out[cl] = per
    return out


# ---------------- gate ----------------
FC, CNT, gate = {}, {}, {}
for v in variants:
    FC[v] = json.loads((OUT / f"out_{v}.geojson").read_bytes())
    CNT[v] = counts(FC[v])
    gate[v] = {"score": {k: M.score_counts(CNT[v], s)[0] for k, s in SL.items()},
               "cls": {c: x["f1"] for c, x in M.score_counts(CNT[v])[1].items()}, "n_out": len(FC[v]["features"])}
    fp_ = CNT[v]["finds_points"]
    def _f(rs):
        tp = sum(fp_[r][0] for r in rs if r in fp_); fp = sum(fp_[r][1] for r in rs if r in fp_); g_ = sum(fp_[r][2] for r in rs if r in fp_)
        return {"tp": tp, "fp": fp, "gt": g_, "f1": 2 * tp / max(2 * tp + fp + g_ - tp, 1)}
    gate[v]["finds"] = {"all": _f(every), "new34": _f(SL["new34"] & every), "nolidar": _f(SL["nolidar"] & every),
                        "fold1_208-217": _f(folds[1]), "fold4_201-207": _f(folds[4]), "fold9_200": _f(folds[9]),
                        "per_region": {r: x for r, x in sorted(fp_.items()) if x[2] or x[0] or x[1]}}
    log(v, "finds", {k: (x["tp"], x["fp"], x["gt"], round(x["f1"], 4)) for k, x in gate[v]["finds"].items() if k != "per_region"})
    log(v, gate[v]["score"])
b = variants[0]
gate["_sanity"] = {"base_all": gate[b]["score"]["all"], "PH_v3": 0.345439, "ok": abs(gate[b]["score"]["all"] - 0.345439) < 5e-4 if b == "base" else None}
for v in variants[1:]:
    g = gate[v]; g["delta"] = {}
    for k, s in SL.items():
        bs = M.paired_bootstrap_counts(CNT[v], CNT[b], D.site_family, 2000, 0, s)
        g["delta"][k] = {"d": g["score"][k] - gate[b]["score"][k], "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    g["p4"] = {c: {"d": g["cls"][c] - gate[b]["cls"][c], "floor": -0.002 if c in TIGHT else -0.005,
                   "ok": g["cls"][c] - gate[b]["cls"][c] >= (-0.002 if c in TIGHT else -0.005) - 1e-12} for c in g["cls"]}
    da = g["delta"]["all"]
    g["checks"] = {"d_all>=0.003": da["d"] >= 0.003, "P>=bar": (da["P"] or 0) >= BONF[v], "P>=0.98333(W1)": (da["P"] or 0) >= 0.98333, "P>=0.985714(info)": (da["P"] or 0) >= 0.985714,
                   "P4": all(x["ok"] for x in g["p4"].values()),
                   "new34>=0": g["delta"]["new34"]["d"] >= 0, "nolidar>=0": g["delta"]["nolidar"]["d"] >= 0}
    g["PASS"] = all(v_ for k_, v_ in g["checks"].items() if k_ not in ("P>=0.98333(W1)", "P>=0.985714(info)"))
    log(v, "delta", {k: round(x["d"], 6) for k, x in g["delta"].items()}, "P", g["delta"]["all"]["P"], "PASS", g["PASS"])
json.dump(gate, open(OUT / ("c1det_gate_vs_" + b + ".json"), "w"), ensure_ascii=False, indent=1, default=float)

