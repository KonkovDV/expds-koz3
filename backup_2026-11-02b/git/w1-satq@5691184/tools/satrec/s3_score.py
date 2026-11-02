#!/usr/bin/env python3
"""W1-SATREC step 3: production-replay gate (PREREG docs/PREREG_W1_SATREC.md 140839a) + satellite recall-loss audit.
argv: variants (first = base). Reads /workspace/koz3/w1/satrec/out/out_<v>.geojson (+ .snap.pkl)."""
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
OUT = Path("/workspace/koz3/w1/satrec/out")
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
folds = [set(f) for f in T["folds"]]; every = set().union(*folds); dreg = set(T["dreg"])
fold_of = {r: i for i, f in enumerate(folds) for r in f}
SL = {"all": None, "new34": set(T["new"]), "nolidar": set(T["optical"])}
gt = {"type": "FeatureCollection", "features": json.loads(Path("/workspace/koz3/w5data/gt_train_v2.geojson").read_bytes())["features"]}
GTR = {cl: {r: v[2] for r, v in d.items()} for cl, d in T["counts_B0"].items()}
TIGHT = {"kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"}
BONF = 1 - 0.1 / 5   # 0.98
SP = ("sp_or", "sp_focal", "sp_arch", "sp_focal2")
variants = ARGV or ["base", "R1", "R2", "R3", "C1", "C2"]


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
    g["checks"] = {"d_all>=0.003": da["d"] >= 0.003, "P>=0.98(bonf)": (da["P"] or 0) >= BONF, "P>=0.9(raw)": (da["P"] or 0) >= 0.9,
                   "P4": all(x["ok"] for x in g["p4"].values()),
                   "new34>=0": g["delta"]["new34"]["d"] >= 0, "nolidar>=0": g["delta"]["nolidar"]["d"] >= 0}
    g["PASS"] = all(v_ for k_, v_ in g["checks"].items() if k_ != "P>=0.9(raw)")
    log(v, "delta", {k: round(x["d"], 6) for k, x in g["delta"].items()}, "P", g["delta"]["all"]["P"], "PASS", g["PASS"])
json.dump(gate, open(OUT / "s3_gate.json", "w"), ensure_ascii=False, indent=1, default=float)

# ---------------- audit ----------------
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
gts = M._collect(tab.mod, tab.calc, S.fc(stk.gt_all), True); M._UTM_GEOM_CACHE.clear()
SAT = {}   # cls -> set of GT idx with markup sp_or, in evaluated regions
for c in tab.CL:
    lst = gts.get(c, [])
    assert len(lst) == len(tab.G[c])
    SAT[c] = {j for j, x in enumerate(lst) if x[3] == "sp_or" and x[2] in every}
GREG = {c: [x[2] for x in gts.get(c, [])] for c in tab.CL}
del gts
nsat = sum(len(v) for v in SAT.values()); log("sat GT", nsat, {c: len(v) for c, v in SAT.items()})


def table(cov):   # cov: set of (cls, j) -> per fold / per class counts
    pf = Counter(fold_of[GREG[c][j]] for c, j in cov); pc = Counter(c for c, _ in cov)
    return {"total": len(cov), "fold": [pf.get(i, 0) for i in range(10)], "cls": dict(pc)}


def covered(feats):
    fs = [{"type": "Feature", "geometry": f["geometry"], "properties": {k: f[k] for k in ("class_name", "confidence", "region_name")}}
          for f in feats]
    cov = set()
    for i in range(0, len(fs), 50000):
        ex = tab._extra(fs[i:i + 50000])
        for c, d in ex.items():
            for _, prs in d["pairs"].items():
                cov.update((c, j) for j, _ in prs if j in SAT[c])
    return cov


A = {"sat_gt": table({(c, j) for c, s in SAT.items() for j in s})}
# L0: raw detector candidates
L0r, L0f = set(), set()
for c, bb in tab.base.items():
    for i, prs in bb["pairs"].items():
        g, r = bb["grp"][i], bb["reg"][i]
        if g in SP and r in dreg:
            tgt = L0f if g == "sp_focal2" else L0r
            tgt.update((c, j) for j, _ in prs if j in SAT[c])
A["L0r_r18"] = table(L0r); A["L0f_f2"] = table(L0f)
raw = []
for p in G.OOF:
    for f, _ in D.iter_features(p):
        q = f["properties"]
        if q.get("class_name") in D.CLASSES5:
            raw.append({"geometry": f["geometry"], "class_name": q["class_name"], "confidence": float(q["confidence"]), "region_name": q["region_name"]})
L0d = covered(raw); del raw
A["L0d_dfine_raw"] = table(L0d); A["L0_union"] = table(L0r | L0f | L0d); A["L0_r18+f2"] = table(L0r | L0f)
log("L0", A["L0r_r18"]["total"], A["L0f_f2"]["total"], A["L0d_dfine_raw"]["total"], A["L0_union"]["total"])
# L1': counterfactual — r18+f2 kept at gate-honest tau (A-stack curves, held fold)
cA, _ = tab.curves(); _, _, ftA = D.honest(cA, folds); del cA
L1c = set()
for c, bb in tab.base.items():
    for i, prs in bb["pairs"].items():
        g, r = bb["grp"][i], bb["reg"][i]
        if g in SP and r in dreg and r in every:
            t = ftA[fold_of[r]].get(g, {}).get(c, math.inf)
            if math.isfinite(t) and bb["conf"][i] >= t:
                L1c.update((c, j) for j, _ in prs if j in SAT[c])
A["L1cf_r18f2_honest_tau"] = table(L1c)


def greedy(fc):   # global greedy like TableEngine.curves; returns assigned (cls, j) per pred, and pred list
    fs = fc["features"]; ex = tab._extra(fs); out = {}
    per = defaultdict(list)
    for f in fs:
        if f["properties"]["class_name"] in ex:
            per[f["properties"]["class_name"]].append(f)
    tp = 0
    for c, d in ex.items():
        assert len(d["conf"]) == len(per[c])
        order = np.argsort(-d["conf"], kind="stable"); used = set(); asg = {}
        for i in order.tolist():
            best, bj = -math.inf, -1
            for j, s_ in sorted(d["pairs"].get(i, [])):
                if j not in used and s_ > best + 1e-9:
                    best, bj = s_, j
            if bj >= 0:
                used.add(bj); asg[i] = bj
        tp += len(asg)
        out[c] = (per[c], asg)
    return out, tp


for v in variants:
    snp = pickle.load(open(OUT / f"out_{v}.geojson.snap.pkl", "rb"))
    st = {}
    for name in ("L1_in", "L2_thr", "L3_thr+rerank", "L3_merge", "L3b_multipole", "L3c_finds", "L3d_twins", "L3e_micro", "L3f_finalize", "L4_final"):
        if name in snp:
            st[name] = table(covered(snp[name])); st[name]["n_pred"] = len(snp[name])
    del snp
    gr, tp = greedy(FC[v])
    tp_off = sum(x[0] for d in CNT[v].values() for x in d.values())
    L5, L5s, nsp, tps = set(), set(), 0, 0
    for c, (fl, asg) in gr.items():
        for i, f in enumerate(fl):
            s_ = f["properties"].get("markup_type") == "sp_or"
            nsp += s_
            if i in asg:
                tps += s_
                if asg[i] in SAT[c]:
                    L5.add((c, asg[i]))
                    if s_:
                        L5s.add((c, asg[i]))
    st["L5_matched_any"] = table(L5); st["L5s_matched_by_sat"] = table(L5s)
    st["sat_pred"] = {"n": nsp, "tp": tps, "precision": tps / max(nsp, 1), "recall_L5s": len(L5s) / nsat, "recall_L5": len(L5) / nsat}
    st["greedy_check"] = {"tp_own": tp, "tp_official": tp_off, "ok": tp == tp_off}
    A[v] = st
    log(v, {k: x["total"] if "total" in x else x for k, x in st.items()})
json.dump(A, open(OUT / "s3_audit.json", "w"), ensure_ascii=False, indent=1, default=float)
log("done")
