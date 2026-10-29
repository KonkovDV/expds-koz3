"""Stream li_hard2_oof_dump.geojson (single 555 MB line) with ijson -> compact npz (region, class, conf, cx, cy, area, nverts)."""
import ijson, sys, numpy as np, json
from shapely.geometry import shape
from collections import Counter
src, dst = sys.argv[1], sys.argv[2]
R, C, P, X, Y, A, NV, MT, TG = [], [], [], [], [], [], [], Counter(), Counter()
regs, clss = {}, {}
with open(src, "rb") as f:
    for ft in ijson.items(f, "features.item", use_float=True):
        pr = ft["properties"]; g = shape(ft["geometry"])
        R.append(regs.setdefault(pr["region_name"], len(regs))); C.append(clss.setdefault(pr["class_name"], len(clss)))
        P.append(pr["confidence"]); c = g.centroid; X.append(c.x); Y.append(c.y); A.append(g.area)
        NV.append(len(ft["geometry"]["coordinates"][0])); MT[pr.get("markup_type")] += 1; TG[pr.get("threshold_group")] += 1
np.savez(dst, region=np.array(R, np.int16), cls=np.array(C, np.int8), conf=np.array(P), x=np.array(X), y=np.array(Y),
         area=np.array(A, np.float32), nv=np.array(NV, np.int32), regions=np.array(list(regs)), classes=np.array(list(clss)))
print(json.dumps({"n": len(P), "regions": len(regs), "classes": list(clss), "markup": MT, "tg": TG}, ensure_ascii=False))
