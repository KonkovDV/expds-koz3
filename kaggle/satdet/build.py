#!/usr/bin/env python3
"""Build the W1-SATDET kernel: embed region→T-fold map (orch/bot-W1/work/v3prod/t_taus.json) into frcnn_src.py."""
import json, sys
from pathlib import Path
mode = sys.argv[1] if len(sys.argv) > 1 else "full"; out = Path(sys.argv[2] if len(sys.argv) > 2 else "koz3-w1-satdet-frcnn.py")
T = json.load(open("/workspace/koz3/orch/bot-W1/work/v3prod/t_taus.json"))
rf = {r: fi for fi, f in enumerate(T["folds"]) for r in f}
assert len(rf) == sum(len(f) for f in T["folds"]) and set(T["dreg"]) <= set(rf), "fold map"
rf = {r: rf[r] for r in sorted(T["dreg"])}  # the 132 SpOr regions
src = (Path(__file__).with_name("frcnn_src.py")).read_text(encoding="utf-8")
out.write_text(src.replace("@@REGION_FOLD@@", json.dumps(rf, ensure_ascii=False, sort_keys=True)).replace("@@MODE@@", mode), encoding="utf-8")
print("wrote", out, mode, len(rf), "regions")
