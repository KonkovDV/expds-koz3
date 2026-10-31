"""Замер: LAS → TIN-DEM 0,1 м → рельеф → кропы 64 px вокруг GT-курганов и li-кандидатов стека (как bot9_kurgan_subclass crops).
Пишет bench_<регион>.npz и строку JSON со временем и пиковой памятью."""
import json, sys, time, resource, logging
from pathlib import Path
import numpy as np
R = Path("/workspace/koz3_shared/bot-9/repo"); sys.path[:0] = [str(R), str(R / "scripts")]
logging.basicConfig(level=logging.WARNING)
import crop_verify as cv
from expds.lidar import RELIEF_TAGS
from expds.catalog import scan_root
root, region, famfile = sys.argv[1], sys.argv[2], sys.argv[3]
K = ("kurgany_povrezhdennye", "kurgany_tselye")
gt = json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))
gfe = [f for f in gt["features"] if f["properties"].get("region_name") == region and f["properties"].get("class_name") in K]
del gt
cand = []
for line in open(famfile, encoding="utf-8"):
    if region not in line: continue
    f = json.loads(line); p = f["properties"]
    if p.get("region_name") == region and p.get("class_name") in K and p.get("markup_type") == "li" and float(p.get("confidence", 0)) >= 0.05:
        cand.append(f)
reg = {r.name: r for r in scan_root(root)}[region]
las_pts = 0
import laspy
for l in reg.las:
    with laspy.open(str(l)) as fh: las_pts += fh.header.point_count
gst = {"features": [{"geometry": f["geometry"], "properties": {"region_name": region}} for f in gfe]}
t0 = time.time(); Xg, okg = cv.extract_crops(root, gst, list(range(len(gfe))), "li", 0.5, 64, tuple(RELIEF_TAGS)); t1 = time.time()
st = {"features": cand}
Xc, okc = cv.extract_crops(root, st, list(range(len(cand))), "li", 0.5, 64, tuple(RELIEF_TAGS)); t2 = time.time()
np.savez_compressed(f"bench_{region}.npz", Xg=Xg[okg], Xc=Xc[okc])
print(json.dumps({"region": region, "las_files": len(reg.las), "las_MB": round(sum(l.stat().st_size for l in reg.las) / 1e6, 1),
                  "las_points": las_pts, "gt_kurgans": len(gfe), "gt_crops": int(okg.sum()), "cand_li_kurgan_conf05": len(cand),
                  "cand_crops": int(okc.sum()), "sec_gt_pass": round(t1 - t0, 1), "sec_cand_pass": round(t2 - t1, 1),
                  "maxrss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, "time_msk": time.strftime("%H:%M:%S")}, ensure_ascii=False))
