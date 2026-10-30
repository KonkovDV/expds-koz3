#!/usr/bin/env python3
"""exec A: retrain pix_clean finds with denser negatives (clean: wraps scripts/pix_clean_train.py, own constants)."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, "/workspace/koz3/wt/w4-pix"); sys.path.insert(0, "/workspace/koz3/wt/w4-pix/scripts")
import pix_clean_train as M
ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--neg-ratio", type=int, default=15)
ap.add_argument("--neg-min", type=int, default=50000); ap.add_argument("--pos-r", type=float, default=0.4)
ap.add_argument("--cache"); ap.add_argument("--out"); ap.add_argument("--exclude", nargs="*", default=[])
ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--mdl", type=int, default=100)
a = ap.parse_args()
M.NEG_RATIO, M.NEG_MIN, M.POS_R_M = a.neg_ratio, a.neg_min, a.pos_r
M.pc.METHODS = ("maggr",)
M.LGB_PARAMS = dict(M.LGB_PARAMS, num_leaves=a.leaves, min_data_in_leaf=a.mdl)
if a.cmd == "sample":
    M.cmd_sample(argparse.Namespace(root="/workspace/koz3/w4geo/train", cache=a.cache, seed=0, regions=None, band_mode="rgb", org_mask=False))
else:
    M.MODEL_FILES = {"finds_points": "finds.txt"}
    M.pc.MODEL_FILES = {"finds_points": "finds.txt"}
    print(M.fit(Path(a.cache), Path(a.out), 0, set(a.exclude), 3, 0, a.rounds)["classes"])
