"""bot-7: сверка порта gor_gate.tif_feats/head_p с признаками bot-5 (feats.npz/meta.json) и p_scene."""
import json, sys, time, resource, importlib.util
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, "/workspace/koz3_shared/bot-7/work6/vendor_test/lib")
sys.path.insert(0, "/workspace/koz3_shared/bot-7/repo")
spec = importlib.util.spec_from_file_location("expds.gor_gate_new", "/workspace/koz3_shared/bot-7/work6/gate/gor_gate.py",
                                              submodule_search_locations=None)
import expds  # noqa
g = importlib.util.module_from_spec(spec); g.__package__ = "expds"; spec.loader.exec_module(g)
B5 = Path("/workspace/koz3_shared/bot-5/analysis/gor_patch")
F5 = np.load(B5 / "feats.npz")["F"].astype(np.float32); M5 = json.load(open(B5 / "meta.json", encoding="utf-8"))
H = g.load_head(B5 / "gor_v3_head.npz", "acb95c52956d712a16c3dd389237a691")
P5 = g.head_p(H, F5)
snap = next(Path("/workspace/koz3_shared/bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots").glob("ed25f*"))
st = dict(g.DEFAULTS, backbone=str(snap), device="cpu")
t = time.time(); m = g.load_backbone(st, Path("/"), torch.device("cpu")); print("load", round(time.time() - t, 1), "s")
D = Path("/workspace/data/train-expds-3"); res = {}
for reg in sys.argv[1:]:
    epsg = int(json.load(open(D / reg / "UTM.json"))["crs"].split(":")[-1])
    tifs = sorted(D.joinpath(reg).glob("*SpOr*/*.tif*")); Fs = []; t = time.time()
    for tf in tifs:
        F, n, ok = g.tif_feats(m, tf, epsg, torch.device("cpu"), 16); Fs.append(F)
    dt = time.time() - t; F = np.concatenate(Fs).astype(np.float32)
    idx = [i for i, x in enumerate(M5) if x["reg"] == reg]
    ref = F5[idx]
    r = dict(n_port=len(F), n_bot5=len(idx), sec=round(dt, 1), sec_per_patch=round(dt / max(1, len(F)), 3))
    if len(F) == len(idx):
        r["max_abs_feat"] = float(np.abs(F.astype(np.float16).astype(np.float32) - ref).max())
        r["cos_min"] = float(((F * ref).sum(1) / np.linalg.norm(F, axis=1) / np.linalg.norm(ref, axis=1)).min())
        p = g.head_p(H, F); r["max_abs_p"] = float(np.abs(p - P5[idx]).max())
        r["p_scene_port"] = float(p.max()); r["p_scene_bot5_insample"] = float(P5[idx].max())
    res[reg] = r; print(reg, r, flush=True)
print("rss_mb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
json.dump(res, open("port_check.json", "w"), ensure_ascii=False, indent=1)
