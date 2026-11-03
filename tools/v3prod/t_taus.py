#!/usr/bin/env python3
"""W1-V3PROD шаг T: воспроизвести гейт W4 (B0, V3P) его же кодом и выгрузить фолды + fold_taus."""
import importlib.util, json, sys, math
from pathlib import Path
_s = importlib.util.spec_from_file_location("v3g", "/workspace/koz3/orch/bot-W4/work/v3_gate_lib.py")
G = importlib.util.module_from_spec(_s); _s.loader.exec_module(G)
D = G.D; M = D.M
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"),
              Path("/workspace/koz3/w1/d3/split_arch"), Path("/workspace/koz3/w1/d3/cache"))
tab = D.TableEngine(Path("/workspace/koz3/w5data/work/table.pkl"), stk)
cA, _ = tab.curves()
regs = {r for c in cA.values() for r in c.pred_region} | {r for c in cA.values() for r in c.gt_regions}
folds = M.family_folds(regs, D.site_family, 10); del cA
no_et = G.drop_engine(tab, lambda g, r: g == "etgr")
c19 = G.load_geo("/workspace/koz3/w4geo/eval/C19etonly_orget_s0.geojson", lambda q: q.get("markup_type") == "etgr")
cB0, _ = no_et.curves(c19); nB0, _, tB0 = D.honest(cB0, folds); del cB0
dfx, dreg = G.dfine_feats()
base3 = G.drop_engine(tab, lambda g, r: g == "etgr" or (g in G.SP and r in dreg)); del tab
cP, _ = base3.curves(c19 + dfx); nP, _, tP = D.honest(cP, folds)
fin = lambda t: {g: {c: (None if not math.isfinite(v) else float(v)) for c, v in d.items()} for g, d in t.items()}
crs = {}
for f in stk.gt_all:
    q = f.get("properties") or {}
    if q.get("original_crs"): crs.setdefault(q["region_name"], q["original_crs"])
out = dict(B0=M.score_counts(nB0)[0], V3P=M.score_counts(nP)[0], folds=[sorted(f) for f in folds], regs=sorted(regs),
           new=sorted(stk.new), optical=sorted(stk.optical), dreg=sorted(dreg), taus_B0=[fin(t) for t in tB0], taus_V3P=[fin(t) for t in tP],
           counts_B0=nB0, counts_V3P=nP, crs=crs)
print("B0", out["B0"], "V3P", out["V3P"], "folds", [len(f) for f in folds])
Path("t_taus.json").write_text(json.dumps(out, ensure_ascii=False))
