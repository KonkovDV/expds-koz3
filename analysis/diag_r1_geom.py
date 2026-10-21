"""Почему пиксели кургана на шаге 1 м не становятся засчитанным объектом.

Для каждого эталона спутника: ближайший предсказанный полигон того же класса,
покрытие меньшей фигуры, лежит ли представительная точка внутри, отношение площадей.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from shapely.geometry import shape
from shapely.strtree import STRtree

sys.path.insert(0, r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")

KEEP = ("kurgany_povrezhdennye", "kurgany_tselye")
PRED = r"F:\expds3\work\spor_r1_oof.geojson"
GT = r"F:\expds3\work\gt_train.geojson"


def coverage(a, b):
    if a.is_empty or b.is_empty or not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    denom = min(a.area, b.area)
    return inter / denom if denom else 0.0


def main() -> None:
    print("читаю", flush=True)
    pred = json.loads(open(PRED, encoding="utf-8").read())["features"]
    gt = json.loads(open(GT, encoding="utf-8").read())["features"]
    by = defaultdict(lambda: defaultdict(list))
    areas = defaultdict(list)
    for f in pred:
        p = f["properties"]
        if p.get("class_name") not in KEEP or float(p.get("confidence", 0)) < 0.5:
            continue
        g = shape(f["geometry"])
        if g.is_empty or g.area <= 0:
            continue
        by[p["region_name"]][p["class_name"]].append((g, float(p["confidence"]), g.area))
        areas[p["class_name"]].append(g.area)
    gt_by = defaultdict(lambda: defaultdict(list))
    gt_areas = defaultdict(list)
    for f in gt:
        p = f["properties"]
        if p.get("markup_type") != "sp_or" or p.get("class_name") not in KEEP:
            continue
        g = shape(f["geometry"])
        gt_by[p["region_name"]][p["class_name"]].append(g)
        gt_areas[p["class_name"]].append(g.area)
    stats = {c: defaultdict(int) for c in KEEP}
    ratios = {c: [] for c in KEEP}
    for region, classes in gt_by.items():
        for cls, geoms in classes.items():
            preds = by[region][cls]
            st = stats[cls]
            st["gt"] += len(geoms)
            if not preds:
                st["no_pred"] += len(geoms)
                continue
            tree = STRtree([g for g, _, _ in preds])
            for g in geoms:
                hits = tree.query(g.buffer(30))
                best_cov, best_in, best_ratio, best_d = 0.0, False, None, 1e9
                cen = g.centroid
                for i in hits:
                    pg, conf, area = preds[int(i)]
                    cov = coverage(pg, g)
                    inside = g.covers(pg.representative_point())
                    d = pg.centroid.distance(cen)
                    if cov > best_cov or (cov == best_cov and d < best_d):
                        best_cov, best_in, best_ratio, best_d = cov, inside, area / g.area if g.area else None, d
                if best_d > 30 and best_cov == 0:
                    # ближайший центр, даже если рамки не пересеклись
                    best_d = min(pg.centroid.distance(cen) for pg, _, _ in preds)
                st["n"] += 1
                if best_d <= 20:
                    st["center20"] += 1
                if best_cov >= 0.5 and best_in:
                    st["match"] += 1
                elif best_cov >= 0.5:
                    st["cov_only"] += 1
                elif best_in and best_cov > 0:
                    st["point_only"] += 1
                if best_ratio is not None and best_d <= 40:
                    ratios[cls].append(best_ratio)
    for cls in KEEP:
        a = np.array(areas[cls]) if areas[cls] else np.array([0.0])
        ga = np.array(gt_areas[cls]) if gt_areas[cls] else np.array([0.0])
        r = np.array(ratios[cls]) if ratios[cls] else np.array([0.0])
        st = stats[cls]
        print(f"\n{cls}")
        print(f"  эталон {st['gt']}, без предсказаний >=0.5 рядом {st['no_pred']}")
        print(f"  центр ближе 20 м {st['center20']}, покрытие и точка {st['match']}, "
              f"только покрытие {st['cov_only']}, только точка {st['point_only']}")
        print(f"  площадь предсказаний p50 {np.percentile(a, 50):.0f} м2, эталон p50 {np.percentile(ga, 50):.0f}")
        if len(r):
            qs = np.percentile(r, [10, 50, 90])
            print(f"  площадь предск/эталон у близких p10/50/90 {qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}")


if __name__ == "__main__":
    main()
