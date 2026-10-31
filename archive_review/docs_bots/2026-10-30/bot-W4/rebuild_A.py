#!/usr/bin/env python3
"""PREREG_w4_compgate R: rebuild OOF stack A on the box from scratch (fresh split + cache), compare to cached A."""
import importlib.util, json, sys, pickle
from pathlib import Path
_s = importlib.util.spec_from_file_location("d3_gate", "/workspace/koz3/orch/bot-W1/work/d3_gate.py")
D = importlib.util.module_from_spec(_s); sys.modules["d3_gate"] = D; _s.loader.exec_module(D)
M, S, log, site_family = D.M, D.S, D.log, D.site_family
CG = Path("/workspace/koz3/w4geo/compgate")
stk = D.Stack(Path("/workspace/koz3/w5data/stack_arch.geojson"), Path("/workspace/koz3/w5data/gt_train_v2.geojson"), CG / "split_arch_R", CG / "cache_R")
old = pickle.load(open(f"/workspace/koz3/w1/d3/cache/curvesA_{stk.key}.pkl", "rb"))
eq = [S.blob(a) == S.blob(b) for a, b in zip(stk.parts, old)]
cR = S.merge(stk.parts); cO = S.merge(old)
regs = {r for c in cO.values() for r in c.pred_region} | {r for c in cO.values() for r in c.gt_regions}
folds = M.family_folds(regs, site_family, 10)
hR = M.score_counts(D.honest(cR, folds)[0])[0]; hO = M.score_counts(D.honest(cO, folds)[0])[0]
res = {"key": stk.key, "clusters": len(stk.parts), "clusters_old": len(old), "blob_equal": sum(eq), "honest_R": hR, "honest_cached": hO,
       "ref": 0.3146153227, "PASS": len(eq) == len(old) and all(eq) and abs(hR - 0.3146153227) <= 1e-9,
       "n_rows": {c: len(v.conf) for c, v in cR.items()}}
log(json.dumps(res, ensure_ascii=False))
(Path(__file__).parent / "rebuild_A.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
