"""На нескольких снимках: какая доля курганов накрыта пиком «центр — окружение».

Радиусы 2/4/7 м — то, что уже считалось на аэрофото. 8/16/28 м — размер насыпи
по эталону спутника (медиана площади около 1000 м², радиус около 18 м).
"""
import json
import sys
from collections import defaultdict

import numpy as np
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.blobs import BlobParams, detect  # noqa: E402
from expds.catalog import scan_root  # noqa: E402
from expds.crs import WEB_MERCATOR, to_crs  # noqa: E402

ROOT = r"F:\expds3\train-expds-3"
GT = r"F:\expds3\work\gt_train.geojson"
CLASSES = ("kurgany_povrezhdennye", "kurgany_tselye")
SCALES = {
    "2/4/7": BlobParams(radii_m=(2.0, 4.0, 7.0), peak_radius_m=4.0, min_contrast=0.2),
    "8/16/28": BlobParams(radii_m=(8.0, 16.0, 28.0), peak_radius_m=12.0, min_contrast=0.15),
}


def main() -> None:
    regs = {r.name: r for r in scan_root(ROOT)}
    gt = json.loads(open(GT, encoding="utf-8").read())["features"]
    by = defaultdict(list)
    for f in gt:
        p = f["properties"]
        if p.get("markup_type") == "sp_or" and p["class_name"] in CLASSES:
            by[p["region_name"]].append(shape(f["geometry"]))
    ranked = sorted(by, key=lambda n: -len(by[n]))
    picked = []
    for name in ranked:
        reg = regs.get(name)
        if reg and reg.rasters.get("sp_or"):
            picked.append(name)
        if len(picked) == 6:
            break
    print("участки:", [(n, len(by[n]), len(regs[n].rasters["sp_or"])) for n in picked], flush=True)
    totals = {k: [0, 0, 0] for k in SCALES}  # gt, hit, candidates
    dists = {k: [] for k in SCALES}
    for name in picked:
        geoms = by[name]
        for path in regs[name].rasters["sp_or"][:2]:
            for key, params in SCALES.items():
                c = detect(path, params)
                n = len(c["x"])
                totals[key][2] += n
                totals[key][0] += len(geoms)
                if n == 0 or c["crs"] is None:
                    dists[key] += [1e9] * len(geoms)
                    print(f"{name} {path.name[-28:]} {key}: кандидатов 0", flush=True)
                    continue
                pts = [Point(x, y) for x, y in zip(c["x"], c["y"])]
                tree = STRtree(pts)
                xy = np.column_stack([c["x"], c["y"]])
                hit = 0
                for g in geoms:
                    gg = to_crs(g, WEB_MERCATOR, c["crs"]) if c["crs"] != WEB_MERCATOR else g
                    inside = tree.query(gg, predicate="intersects")
                    cen = gg.centroid
                    d = np.hypot(xy[:, 0] - cen.x, xy[:, 1] - cen.y).min()
                    dists[key].append(float(d))
                    hit += int(len(inside) > 0)
                totals[key][1] += hit
                print(f"{name} {path.name[-28:]} {key}: кандидатов {n}, "
                      f"курганов с пиком внутри {hit}/{len(geoms)}", flush=True)
    print("---", flush=True)
    for key, (ng, nh, nc) in totals.items():
        d = np.array(dists[key])
        d = d[d < 1e8]
        qs = np.percentile(d, [10, 50, 90]) if len(d) else [0, 0, 0]
        print(f"{key}: пик внутри {nh}/{ng} = {nh / ng if ng else 0:.3f}, кандидатов {nc}, "
              f"дистанция до центра p10/50/90 {qs[0]:.0f}/{qs[1]:.0f}/{qs[2]:.0f} м", flush=True)


if __name__ == "__main__":
    main()
