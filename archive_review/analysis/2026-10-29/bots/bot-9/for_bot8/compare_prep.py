"""Сравнение выхода cnx_prep с эталоном prep-ядра (_scratch/relief/*.npy = выход кода koz3-cnx-li-prep, dry-run)."""
import json, sys
import numpy as np
REF = "../kaggle/convnext_lidar/_scratch/relief"
res = {}
for f in sys.argv[1:]:
    tag, reg = f.split("/")[-1][:-4].split("_", 1)
    z = np.load(f); u8 = z["u8"]; geo = json.loads(str(z["geo"]))
    ref = np.load(f"{REF}/{reg}.npy"); meta = json.load(open(f"{REF}/{reg}.json"))
    out = dict(shape_eq=list(u8.shape) == list(ref.shape), left_eq=geo["left"] == meta["left"], top_eq=geo["top"] == meta["top"],
               stats_eq=geo["site_stats"] == meta["site_stats"])
    if out["shape_eq"]:
        for ch in (3, 5):
            a = (u8[:ch].astype(np.float32) - 128) / 127; b = (ref[:ch].astype(np.float32) - 128) / 127
            out[f"c{ch}_max_abs"] = float(np.abs(a - b).max()); out[f"c{ch}_u8_ndiff"] = int((u8[:ch] != ref[:ch]).sum())
    res[f"{tag}:{reg}"] = out
print(json.dumps(res, ensure_ascii=False, indent=1))
