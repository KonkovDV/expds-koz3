"""Паритет признаков expds/rerank_arch.py с X2 реранкера (/workspace/rerank/work/X2.npy) на строках архитектуры
и проверка адаптера table() (3857 → UTM) на кандидатах архитектуры из stack_arch.geojson."""
import importlib.util, itertools, json, sys
from pathlib import Path
import numpy as np
W = Path("/workspace/rerank"); sys.path.insert(0, str(W))
import fast
spec = importlib.util.spec_from_file_location("expds.rerank_arch", Path("repo/expds/rerank_arch.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
T = dict(fast.T); cls = T["cls"].astype(int); A = R.METRIC_CLASSES.index("arkhitektury")
boxes = json.loads(Path("/workspace/expds-koz3/notes/las_boxes_59.json").read_text(encoding="utf-8"))
T["has_las"] = np.isin(T["reg"], list(boxes)); T["twin"] = np.load(W / "work/twin.npy")
target = cls == A
X2 = np.load(W / "work/X2.npy", mmap_mode="r")
meta = json.load(open("work3/rerank_arch_lgbm.json"))
Xm = R.features(T, target, meta["grp_vocab"], meta["mk_vocab"])
Xr = np.asarray(X2[np.flatnonzero(target)])
out = {"rows": int(target.sum())}
for j, n in enumerate(R.NAMES):
    d = np.abs(Xm[:, j].astype(np.float64) - Xr[:, j].astype(np.float64))
    out[n] = [float(d.max()), int((d > 1e-5).sum())]
pool = R.pool_mask(T["grp"], cls, T["conf"]); out["pool_equal"] = bool((pool == np.load(W / "work/pool.npy")).all())
print(json.dumps(out, ensure_ascii=False))
# адаптер: первые 4000 кандидатов архитектуры из стека (порядок файла = порядок table.pkl)
import ijson, shapely
from shapely.geometry import shape
feats, rows = [], []
with open(W / "stack_arch.geojson", "rb") as fh:
    for i, f in enumerate(ijson.items(fh, "features.item", use_float=True)):
        if cls[i] == A:
            p = f["properties"]; feats.append(dict(p, geom=shape(f["geometry"]))); rows.append(i)
            if len(feats) >= 4000: break
rows = np.array(rows); t = R.table(feats)
ad = {}
for k in ["area", "perim", "bw", "bh"]:
    rel = np.abs(t[k] - T[k][rows]) / np.maximum(np.abs(T[k][rows]), 1e-9); ad[k] = [float(np.median(rel)), float(rel.max())]
for k in ["cx", "cy"]:
    ad[k] = float(np.abs(t[k] - T[k][rows]).max())
ad["nv_equal"] = bool((t["nv"] == T["nv"][rows]).all()); ad["conf_max_abs"] = float(np.abs(t["conf"] - T["conf"][rows]).max())
ad["grp_equal"] = bool((t["grp"] == T["grp"][rows]).all()); ad["mk_equal"] = bool((t["mk"] == T["mk"][rows]).all())
ad["reg_equal"] = bool((t["reg"] == T["reg"][rows]).all()); ad["cls_equal"] = bool((t["cls"] == cls[rows]).all())
ad["n"] = len(rows); ad["last_row"] = int(rows[-1])
print(json.dumps(ad, ensure_ascii=False))
json.dump({"features": out, "adapter": ad}, open("work3/parity.json", "w"), ensure_ascii=False, indent=1)
