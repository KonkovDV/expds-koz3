"""Дефект цели карты центров: касающиеся на сетке 0,5 м эталоны сливаются в одну компоненту (centers.gaussians_from_binary),
пик — в центре масс слитой компоненты; find_peaks с min_dist_m=4 гасит близкие пики. Оракул: один круг на компоненту."""
import json, sys, io, contextlib, math
from collections import defaultdict, Counter
from shapely.ops import unary_union
from shapely.geometry import Point
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials"); sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import compute_metrics as cm
from expds.naming import site_family
gt = json.load(open("gt_arch_3857.geojson"))
by = defaultdict(list)
for f in gt["features"]:
    g = cm._extract_geometry(f)
    if g is None: continue
    p = f["properties"]; by[(p["region_name"], p["markup_type"])].append(cm._to_utm(g, "auto"))
calc = cm.F1Calculator()
out = {}
for res in (0.5,):
    agg = defaultdict(lambda: Counter())
    for (reg, mt), gs in by.items():
        # компоненты: зазор < res (пиксели касаются)
        u = unary_union([g.buffer(res / 2) for g in gs])
        comps = list(getattr(u, "geoms", [u]))
        peaks = []
        for c in comps:
            members = [g for g in gs if g.intersects(c)]
            cen = unary_union(members).centroid
            peaks.append((cen, len(members)))
        # подавление пиков ближе 4 м (порядок: крупнее раньше)
        peaks.sort(key=lambda t: -t[1])
        kept = []
        for cen, n in peaks:
            if all(cen.distance(k) >= 4.0 for k in kept): kept.append(cen)
        for rad in (2.0, 0.05):
            used = set(); tp = 0
            for cen in kept:
                circ = cen.buffer(rad, quad_segs=16)
                for j, g in enumerate(gs):
                    if j in used: continue
                    if calc._match_polygon_gt(circ, g)[0]:
                        used.add(j); tp += 1; break
            agg[(mt, rad)]["tp"] += tp
            agg[("ALL", rad)]["tp"] += tp
        for key in (mt, "ALL"):
            agg[(key, "n")]["gt"] += len(gs); agg[(key, "n")]["comps"] += len(comps); agg[(key, "n")]["peaks_after_nms"] += len(kept)
            agg[(key, "n")]["gt_in_multi_comps"] += sum(1 for c in comps if sum(g.intersects(c) for g in gs) > 1 for _ in [0]) and sum(sum(g.intersects(c) for g in gs) for c in comps if sum(g.intersects(c) for g in gs) > 1)
    for k, v in agg.items(): out[f"{k[0]}|{k[1]}"] = dict(v)
for k in sorted(out): print(k, out[k])
json.dump(out, open("../analysis/arch_component_oracle.json", "w"), ensure_ascii=False, indent=1)
