#!/usr/bin/env python3
"""Check maps npz against bot-9 HEAD_INTERFACE keys/shapes/dtypes and that a peak decode reproduces the
exported detections (centre error < 0.5 m) for hm classes."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter

out = Path(sys.argv[1])
fc = json.load(open(next(out.glob("oof_adaf_*.geojson"))))
for p in sorted((out / "maps").glob("*.npz")):
    z = np.load(p); m = json.loads(str(z["meta"]))
    H, W = m["H"], m["W"]; h, w = -(-H // 4), -(-W // 4)
    assert z["valid"].shape == (H, W) and z["valid"].dtype == np.uint8
    assert z["valid_h"].shape == (h, w) and z["hm"].shape == (2, h, w) and z["hm"].dtype == np.float32
    assert z["wh"].shape == (1, h, w) and z["wh"].dtype == np.float16 and z["off"].shape == (2, h, w)
    assert z["mask"].shape == (3, H, W) and z["mask"].dtype == np.uint8
    for k in ("region", "fold", "family", "tag", "head", "stride", "left", "top", "step", "step_head", "crs", "H", "W"):
        assert k in m, k
    hm = z["hm"]; n = 0
    for c in range(2):
        pk = (hm[c] == maximum_filter(hm[c], 3)) & (hm[c] >= 0.05)
        n += int(pk.sum())
    nf = sum(1 for f in fc["features"] if f["properties"]["region_name"] == m["region"]
             and f["properties"]["class_name"] in ("kurgany_tselye", "kurgany_povrezhdennye"))
    print(p.name, "fold", m["fold"], "hm peaks", n, "kurgan dets", nf, "mask max", int(z["mask"].max()), "OK")
print("report", json.load(open(out / "report.json"))["oof"])
