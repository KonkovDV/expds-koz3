"""Доля курганов, у которых сильный пик лежит у центра, при одинаковом бюджете пиков.

«Пик внутри полигона» на мелком масштабе завышен: пиков десятки тысяч, и крупный
полигон ловит случайный. Здесь остаются K самых сильных пиков снимка.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from shapely.geometry import Point, shape

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
from expds.blobs import BlobParams, detect  # noqa: E402
from expds.catalog import scan_root  # noqa: E402
from expds.crs import WEB_MERCATOR, to_crs  # noqa: E402

NAMES = ["163_Курганы_34", "048_Калмыкия_1", "131_Курганы_2", "057_Кабардино_Балкария", "047_Тыва_2"]
SCALES = {
    "2/4/7": BlobParams(radii_m=(2.0, 4.0, 7.0), peak_radius_m=4.0, min_contrast=0.2),
    "8/16/28": BlobParams(radii_m=(8.0, 16.0, 28.0), peak_radius_m=12.0, min_contrast=0.15),
    "4/10/20": BlobParams(radii_m=(4.0, 10.0, 20.0), peak_radius_m=8.0, min_contrast=0.15),
}
KS = (100, 300, 1000)


def main() -> None:
    regs = {r.name: r for r in scan_root(r"F:\expds3\train-expds-3")}
    gt = json.loads(open(r"F:\expds3\work\gt_train.geojson", encoding="utf-8").read())["features"]
    by = defaultdict(list)
    for f in gt:
        p = f["properties"]
        if p.get("markup_type") == "sp_or" and p["class_name"] in ("kurgany_povrezhdennye", "kurgany_tselye"):
            by[p["region_name"]].append(shape(f["geometry"]))
    for name in NAMES:
        path = regs[name].rasters["sp_or"][0]
        geoms = by[name]
        print(f"\n{name} курганов {len(geoms)} {path.name}", flush=True)
        for key, params in SCALES.items():
            c = detect(path, params)
            if len(c["x"]) == 0:
                print(f"  {key}: нет пиков", flush=True)
                continue
            strength = c["features"][:, len(params.radii_m)]
            order = np.argsort(-strength)
            cents = []
            for g in geoms:
                gg = to_crs(g, WEB_MERCATOR, c["crs"]) if c["crs"] != WEB_MERCATOR else g
                cents.append((gg.centroid.x, gg.centroid.y, gg))
            xy_all = np.column_stack([c["x"], c["y"]])
            bits = []
            for k in KS:
                idx = order[: min(k, len(order))]
                xy = xy_all[idx]
                near = inside = 0
                for x, y, gg in cents:
                    d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
                    near += int(d.min() <= 15.0)
                    # точка внутри полигона — то, что увидит метка LightGBM
                    hit = False
                    j = int(np.argmin(d))
                    if d[j] <= 80:
                        hit = gg.covers(Point(xy[j, 0], xy[j, 1])) or gg.intersects(Point(xy[j, 0], xy[j, 1]))
                    # точнее: любой из K, но только если ближайший далеко не проверять все — проверим ближайшие 8
                    if not hit:
                        for j in np.argsort(d)[:8]:
                            if d[j] > 80:
                                break
                            if gg.intersects(Point(float(xy[j, 0]), float(xy[j, 1]))):
                                hit = True
                                break
                    inside += int(hit)
                bits.append(f"K={k}: центр≤15м {near}/{len(cents)}, внутри {inside}/{len(cents)}")
            print(f"  {key} пиков {len(order)}  " + " | ".join(bits), flush=True)


if __name__ == "__main__":
    main()
