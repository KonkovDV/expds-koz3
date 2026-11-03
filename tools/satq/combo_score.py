#!/usr/bin/env python3
"""W1-COMBO gate (PREREG docs/PREREG_W1_COMBO.md): COMBO vs base (decision, P >= 0.98333), info: COMBO vs C1F alone and vs FP1 alone.
base/C1F/FP1 = existing replays (frzq/out_base, frz/out_C1F, frzq/out_FP1; read only)."""
import importlib.util, json, math, pickle, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

ARGV = sys.argv[1:]; sys.argv = sys.argv[:1]
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D = G.D; M = D.M; S = D.S
OUT = Path("/workspace/koz3/w1/satq/combo")
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
folds = [set(f) for f in T["folds"]]; every = set().union(*folds); dreg = set(T["dreg"])
fold_of = {r: i for i, f in enumerate(folds) for r in f}
SL = {"all": None, "new34": set(T["new"]), "nolidar": set(T["optical"])}
gt = {"type": "FeatureCollection", "features": json.loads(Path("/workspace/koz3/w5data/gt_train_v2.geojson").read_bytes())["features"]}
GTR = {cl: {r: v[2] for r, v in d.items()} for cl, d in T["counts_B0"].items()}
TIGHT = {"kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"}
BONF = 1 - 0.05 / 3   # 0.98333, kept = FRZ (PREREG COMBO; m = 1)
PTHR = lambda v: BONF
SP = ("sp_or", "sp_focal", "sp_arch", "sp_focal2")
variants = ["base", "COMBO"]
PATH = {"base": "/workspace/koz3/w1/satq/frzq/out_base.geojson", "C1F": "/workspace/koz3/w1/satq/frz/out_C1F.geojson",
        "FP1": "/workspace/koz3/w1/satq/frzq/out_FP1.geojson", "COMBO": "/workspace/koz3/w1/satq/combo/out_COMBO.geojson"}


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
for v in ["base", "C1F", "FP1", "COMBO"]:
    FC[v] = json.loads(Path(PATH[v]).read_bytes())
    CNT[v] = counts(FC[v])
    gate[v] = {"score": {k: M.score_counts(CNT[v], s)[0] for k, s in SL.items()},
               "cls": {c: x["f1"] for c, x in M.score_counts(CNT[v])[1].items()}, "n_out": len(FC[v]["features"])}
    log(v, gate[v]["score"])
b = variants[0]
gate["_sanity"] = {"base_all": gate[b]["score"]["all"], "PH_v3": 0.345439, "ok": abs(gate[b]["score"]["all"] - 0.345439) < 5e-4}
for v in variants[1:]:
    g = gate[v]; g["delta"] = {}
    for k, s in SL.items():
        bs = M.paired_bootstrap_counts(CNT[v], CNT[b], D.site_family, 2000, 0, s)
        g["delta"][k] = {"d": g["score"][k] - gate[b]["score"][k], "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
    g["p4"] = {c: {"d": g["cls"][c] - gate[b]["cls"][c], "floor": -0.002 if c in TIGHT else -0.005,
                   "ok": g["cls"][c] - gate[b]["cls"][c] >= (-0.002 if c in TIGHT else -0.005) - 1e-12} for c in g["cls"]}
    da = g["delta"]["all"]
    g["checks"] = {"d_all>=0.003": da["d"] >= 0.003, "P>=thr": (da["P"] or 0) >= PTHR(v), "P>=0.9(raw)": (da["P"] or 0) >= 0.9,
                   "P4": all(x["ok"] for x in g["p4"].values()),
                   "new34>=0": g["delta"]["new34"]["d"] >= 0, "nolidar>=0": g["delta"]["nolidar"]["d"] >= 0}
    g["P_thr"] = PTHR(v); g["PASS"] = all(v_ for k_, v_ in g["checks"].items() if k_ != "P>=0.9(raw)")
    log(v, "delta", {k: round(x["d"], 6) for k, x in g["delta"].items()}, "P", g["delta"]["all"]["P"], "PASS", g["PASS"])
for ref in ("C1F", "FP1"):   # information only
    gate["COMBO"]["vs_" + ref] = {}
    for k, s in SL.items():
        bs = M.paired_bootstrap_counts(CNT["COMBO"], CNT[ref], D.site_family, 2000, 0, s)
        gate["COMBO"]["vs_" + ref][k] = {"d": gate["COMBO"]["score"][k] - gate[ref]["score"][k], "P": bs.get("p_gt0")}
    log("COMBO vs", ref, {k: (round(x["d"], 6), x["P"]) for k, x in gate["COMBO"]["vs_" + ref].items()})
json.dump(gate, open(OUT / "s3_gate_COMBO.json", "w"), ensure_ascii=False, indent=1, default=float)

log("done")
