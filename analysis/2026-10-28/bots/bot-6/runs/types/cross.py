"""Сколько объектов эталона — «второй слой» того же объекта в другом типе данных (правило метрики)."""
import json, sys, collections
sys.path.insert(0, "repo/materials"); sys.path.insert(0, "repo")
from shapely.geometry import shape
from shapely import STRtree
import compute_metrics as cm
calc = cm.F1Calculator() if hasattr(cm, "F1Calculator") else None
fc = json.load(open("work/types/gt_repo.geojson", encoding="utf-8"))
by = collections.defaultdict(list)
for f in fc["features"]:
    p = f["properties"]
    g = cm._to_utm(shape(f["geometry"]), "auto")
    by[(p["region_name"], p["class_name"])].append((p["markup_type"], g))
res = collections.Counter(); tot = collections.Counter()
for (reg, cls), items in by.items():
    polys = [(mt, g) for mt, g in items if g.geom_type != "Point"]
    if not polys: 
        for mt, g in items: tot[(cls, mt)] += 1
        continue
    tree = STRtree([g for _, g in polys])
    for i, (mt, g) in enumerate(polys):
        tot[(cls, mt)] += 1
        hit_types = set()
        for j in tree.query(g):
            mt2, g2 = polys[j]
            if mt2 == mt: continue
            ok, _ = calc._match_polygon_gt(g2, g)   # копия чужого эталона как предсказание
            if ok: hit_types.add(mt2)
        if hit_types: res[(cls, mt)] += 1
rows = [{"class": c, "type": m, "gt": tot[(c, m)], "hit_by_other_type_gt": res[(c, m)]} for (c, m) in sorted(tot)]
for r in rows: print(r)
print("ИТОГО", sum(tot.values()), "из них покрываются эталоном другого типа", sum(res.values()))
json.dump(rows, open("work/types/cross_type_gt.json", "w"), ensure_ascii=False, indent=1)
