#!/usr/bin/env python3
"""KURGAN-DUAL gate (prereg/KURGAN_DUAL_PREREG.md). Clean-room from spec. V3P context as W4 compgate (d3_gate/v3_gate_lib unchanged)."""
import importlib.util, json, sys, gc, pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D, M, log, site_family = G.D, G.M, G.log, G.site_family
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = Path("/workspace/koz3/orch/bot-W3/runs/gate_KDUAL_DMG.json")
PO, TS = "kurgany_povrezhdennye", "kurgany_tselye"; KC = (PO, TS); LIG = {"li", "li_dev", "li_mosaic"}
GRID = [None, 0.20, 0.15, 0.10, 0.075, 0.05, 0.03, 0.02, 0.01]   # None = off; order = tie preference (larger first)
res = {"boot": BOOT, "sanity": {}}
def save(): OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
TP = Path("/workspace/koz3/w5data/work/table.pkl")
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(TP, stk)
slices = {"all": None, "new34": stk.new, "nolidar": stk.optical}
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10); every = set().union(*folds); del cA; gc.collect()
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
dfx, dreg = G.dfine_feats(); SP = G.SP
drop = lambda g, r: g == "etgr" or (g in SP and r in dreg)
b3 = G.drop_engine(tab, drop)
cP, _ = b3.curves(c19 + dfx)
nB, info, taus = D.honest(cP, folds)
sB = M.score_counts(nB)[0]; res["sanity"]["V3P"] = sB; log(f"V3P honest {sB:.6f}"); save()
# row-level hit flags for kurgan classes (same matching as TableEngine.curves)
ex = b3._extra(c19 + dfx); rowhit = {}
for c in KC:
    b, e = b3.base[c], ex.get(c); nb = len(b["conf"])
    conf = b["conf"] if e is None else np.concatenate([b["conf"], e["conf"]])
    pairs = dict(b["pairs"])
    if e is not None: pairs.update({nb + i: v for i, v in e["pairs"].items()})
    order = np.argsort(-conf, kind="stable"); rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))
    hit = np.zeros(len(conf), bool); used = set()
    for i in sorted(pairs, key=lambda i: rank[i]):
        best, bj = float("-inf"), -1
        for j, s_ in sorted(pairs[i]):
            if j not in used and s_ > best + 1e-9: best, bj = s_, j
        if bj >= 0: used.add(bj); hit[i] = True
    assert hit.sum() == cP[c].hit.sum(); rowhit[c] = hit[:nb]
del ex; gc.collect()
# pair li copies by identical geometry (table cx, cy, area, nv)
with open(TP, "rb") as fh: T = pickle.load(fh)
pos = {}
for k, c in enumerate(T["CL"]):
    if c not in KC: continue
    idx = np.flatnonzero(T["cls"] == k)
    keep = np.array([not drop(g, r) for g, r in zip(tab.base[c]["grp"], tab.base[c]["reg"])], bool)
    gi = idx[keep]; assert len(gi) == len(b3.base[c]["conf"])
    pos[c] = {(T["reg"][i], round(float(T["cx"][i]), 3), round(float(T["cy"][i]), 3), round(float(T["area"][i]), 3), int(T["nv"][i]), str(T["grp"][i])): p
              for p, i in enumerate(gi) if T["grp"][i] in LIG}
del T, tab; gc.collect()
keys = set(pos[PO]) & set(pos[TS])
res["sanity"]["li_pairs"] = len(keys); res["sanity"]["li_unpaired"] = [len(pos[PO]) - len(keys), len(pos[TS]) - len(keys)]
S = []   # second-class copies: (cls, other, g, reg, conf, conf_other, hit)
for kk in keys:
    p, t = pos[PO][kk], pos[TS][kk]
    cp, ct = b3.base[PO]["conf"][p], b3.base[TS]["conf"][t]
    g = str(b3.base[PO]["grp"][p]); assert g == str(b3.base[TS]["grp"][t])
    if cp > ct: S.append((TS, PO, g, kk[0], ct, cp, bool(rowhit[TS][t])))
    else: S.append((PO, TS, g, kk[0], cp, ct, bool(rowhit[PO][p])))
S = [s for s in S if s[0] == PO]  # KDUAL-dmg: only damaged second copies (flag kurgan_dual_dmg)
log(f"li pairs {len(keys)} unpaired {res['sanity']['li_unpaired']}"); save()
def adds(tk, tau, regset):
    out = {c: defaultdict(lambda: [0, 0, 0]) for c in KC}
    if tau is None: return out
    for c, o, g, r, x, xo, h in S:
        if r not in regset or x < tau: continue
        tc, to = tk.get(g, {}).get(c, np.inf), tk.get(g, {}).get(o)
        if to is None or xo < to or x >= tc: continue
        out[c][r][0 if h else 1] += 1
    return out
def base_counts(tk, regset):
    out = {}
    for c in KC:
        cv = cP[c]; per = {r: [0, 0, int(k)] for r, k in cv.gt_regions.items() if r in regset}
        for x, h, r, g in zip(cv.conf, cv.hit, cv.pred_region, cv.pred_group):
            if r in regset and x >= tk.get(g, {}).get(c, np.inf):
                l = per.setdefault(r, [0, 0, 0]); l[0 if h else 1] += 1
        out[c] = per
    return out
nD = {c: {r: list(v) for r, v in by.items()} for c, by in nB.items()}; chosen = []
for k, held in enumerate(folds):
    tk = taus[k]; train = every - held; bt = base_counts(tk, train); best, bs = None, -1
    for tau in GRID:
        a = adds(tk, tau, train); cnt = {c: {r: list(v) for r, v in bt[c].items()} for c in KC}
        for c in KC:
            for r, v in a[c].items(): l = cnt[c].setdefault(r, [0, 0, 0]); l[0] += v[0]; l[1] += v[1]
        s = M.score_counts(cnt)[0]
        if s > bs + 1e-12: best, bs = tau, s
    chosen.append(best)
    for c in KC:
        for r, v in adds(tk, best, held)[c].items(): l = nD[c].setdefault(r, [0, 0, 0]); l[0] += v[0]; l[1] += v[1]
log(f"tau_dual per fold {chosen}")
out = {"tau_dual": chosen}
for nm, s in slices.items():
    a, b = M.score_counts(nB, s)[0], M.score_counts(nD, s)[0]; bs = M.paired_bootstrap_counts(nD, nB, site_family, BOOT, 0, s)
    out[nm] = {"B": a, "DUAL": b, "delta": b - a, "P": bs.get("p_gt0"), "p5": bs.get("p5"), "p95": bs.get("p95")}
pa, pb = M.score_counts(nB)[1], M.score_counts(nD)[1]
out["cls"] = {c: {"d": pb[c]["f1"] - pa[c]["f1"], "ok": pb[c]["f1"] - pa[c]["f1"] >= -(0.002 if c in D.FLOOR_TIGHT else 0.005) - 1e-12} for c in pa if pa[c]["n_gt"] > 0}
g = {"delta": out["all"]["delta"] >= 0.003, "P": (out["all"]["P"] or 0) >= 0.9, "new34": out["new34"]["delta"] >= 0, "nolidar": out["nolidar"]["delta"] >= -1e-12,
     "damaged": out["cls"][PO]["d"] >= 0, "fort": out["cls"]["fortifikatsii"]["d"] >= -1e-12, "floors": all(v["ok"] for v in out["cls"].values())}
g["PASS"] = all(g.values()); out["gate"] = g; res["gate"] = out; save()
log(f"DUAL vs V3P: Δ {out['all']['delta']:+.6f} P {out['all']['P']} | new34 {out['new34']['delta']:+.6f} nolidar {out['nolidar']['delta']:+.6f} | "
    + " ".join(f"{c[:8]} {v['d']:+.5f}" for c, v in out["cls"].items()) + f" | {g}")
