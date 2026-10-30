#!/usr/bin/env python3
"""exec A: candidate tag = src decode tag minus maggr object_poly pix (classic poly stays), optional baked union
of MagGr finds (classic finds kept + pix finds not within 1 m / same as pipeline.dedupe_union), replace for etgr."""
import sys, pickle, argparse
from pathlib import Path
sys.path.insert(0, "/workspace/koz3/wt/w4-pix")
from expds import pix_clean as pc
W = Path("/workspace/koz3/w4geo/work")
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--union", action="store_true")
a = ap.parse_args()
od = W / "cache/pix" / a.dst; od.mkdir(parents=True, exist_ok=True)
for f in sorted((W / "cache/pix" / a.src).glob("*.pkl")):
    rec = pickle.loads(f.read_bytes())
    new = [x for x in rec["new"] if not (x["class_name"] == "object_poly" and x["markup_type"] == "maggr")]
    pst = {k: {kk: vv for kk, vv in v.items() if kk != ("object_poly", "maggr")} for k, v in rec["pst"].items()}
    if a.union:
        cl = [x for x in pickle.loads((W / "cache/classic" / f.name).read_bytes())
              if x["class_name"] == "finds_points" and x["markup_type"] == "maggr"]
        pf = [x for x in new if x["class_name"] == "finds_points" and x["markup_type"] == "maggr"]
        if pf:
            keep = pc.dedupe_union(cl, pf)
            new = [x for x in new if not (x["class_name"] == "finds_points" and x["markup_type"] == "maggr")] + cl + keep
    (od / f.name).write_bytes(pickle.dumps({"new": new, "pst": pst, "model_dir": rec.get("model_dir")}))
print("ok", a.dst)
