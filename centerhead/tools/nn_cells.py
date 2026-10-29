"""Доля Li-курганов, у которых ближайший сосед ТОГО ЖЕ класса ближе d м (риск слияния пиков на сетке stride·0,5 м)."""
import sys, json, collections
import numpy as np
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo/materials")
import compute_metrics as CM
from shapely.geometry import shape
from scipy.spatial import cKDTree
d = json.load(open("/workspace/rerank/gt_train_v2.geojson"))
P = collections.defaultdict(list)
for f in d["features"]:
    p = f["properties"]
    if p.get("markup_type") == "li" and p.get("class_name") in ("kurgany_tselye", "kurgany_povrezhdennye"):
        g = CM._to_utm(shape(f["geometry"]), "auto"); c = g.centroid
        if not g.contains(c): c = g.representative_point()
        P[(p["class_name"], p.get("region_name"))].append((c.x, c.y))
out = {}
for cls in ("kurgany_tselye", "kurgany_povrezhdennye"):
    nn = []
    for (c, r), pts in P.items():
        if c != cls or len(pts) < 2: continue
        a = np.array(pts); dd, _ = cKDTree(a).query(a, k=2); nn += dd[:, 1].tolist()
    nn = np.array(nn); n = sum(len(v) for (c, _), v in P.items() if c == cls)
    out[cls] = {"n": n, **{f"frac_nn_same_cls_lt_{t}m": round(float((nn < t).sum() / n), 4) for t in (1, 2, 3, 4)}}
print(json.dumps(out, ensure_ascii=False))
json.dump(out, open("/workspace/koz3_shared/bot-2/work/centerhead/nn_cells.json", "w"), indent=1)
