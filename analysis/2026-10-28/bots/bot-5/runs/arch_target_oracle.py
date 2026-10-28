"""Потолок полноты идеальной карты центров при разных целях обучения (официальное попадание compute_metrics).
Цель = центр масс связной компоненты растра 0,5 м (all_touched ≈ зазор < 0,5 м сливается), пики не ближе min_dist."""
import json, sys, io, contextlib, math
from collections import defaultdict, Counter
from shapely.ops import unary_union
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import compute_metrics as cm

def shrink_polygon(g, shrink):  # копия expds/tiles.py:161
    if shrink <= 0 or g.geom_type == "Point" or g.area <= 0:
        return g
    r = math.sqrt(g.area / math.pi)
    s = g.buffer(-shrink * r)
    if s.is_empty or s.area < 0.05 * g.area:
        s = g.representative_point().buffer(max(r * (1 - shrink), 0.5))
    return s

gt = json.load(open("gt_arch_3857.geojson"))
by = defaultdict(list)
for f in gt["features"]:
    g = cm._extract_geometry(f)
    if g is None: continue
    p = f["properties"]; by[(p["region_name"], p["markup_type"])].append(cm._to_utm(g, "auto"))
calc = cm.F1Calculator()
res = {}
for shrink in (0.0, 0.25, 0.4, 0.6):
    for md in (4.0, 2.0):
        for rad in (2.0, 0.05):
            agg = Counter()
            for (reg, mt), gs in by.items():
                tg = [shrink_polygon(g, shrink) for g in gs]
                u = unary_union([t.buffer(0.25) for t in tg])
                comps = list(getattr(u, "geoms", [u]))
                peaks = []
                for c in comps:
                    mem = [t for t in tg if t.intersects(c)]
                    peaks.append((unary_union(mem).centroid, sum(t.area for t in mem)))
                peaks.sort(key=lambda t: -t[1])
                kept = []
                for cen, _ in peaks:
                    if all(cen.distance(k) >= md for k in kept): kept.append(cen)
                used = set(); tp = 0
                for cen in kept:
                    circ = cen.buffer(rad, quad_segs=16)
                    best = -1; bs = -1
                    for j, g in enumerate(gs):
                        if j in used: continue
                        h, s = calc._match_polygon_gt(circ, g)
                        if h and s > bs: best, bs = j, s
                    if best >= 0: used.add(best); tp += 1
                for key in (mt, "ALL"):
                    agg[key + "_tp"] += tp; agg[key + "_gt"] += len(gs)
            k = f"shrink={shrink} min_dist={md} r={rad}"
            res[k] = {m: f'{agg[m+"_tp"]}/{agg[m+"_gt"]}={agg[m+"_tp"]/agg[m+"_gt"]:.3f}' for m in ("ALL", "sp_or", "or", "ae")}
            print(k, res[k], flush=True)
json.dump(res, open("../analysis/arch_target_oracle.json", "w"), ensure_ascii=False, indent=1)
