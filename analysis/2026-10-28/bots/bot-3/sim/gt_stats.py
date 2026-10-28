"""Статистика эталона геофизики: где метрика теряет баллы на склейке и геометрии (только эталон)."""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from collections import defaultdict
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
from shapely.ops import polylabel
from load_gt import load

U = load()
out = {}
# --- finds_points: соседи ближе r, потолок полноты при подавлении пиков радиусом r
fp = defaultdict(list)
for u in U:
    if u["cls"] == "finds_points": fp[(u["family"], u["region"])].append(np.array([[g.x, g.y] for g in u["geoms"]]))
rng = np.random.default_rng(0)
res = defaultdict(lambda: defaultdict(float))
for (fam, reg), arrs in fp.items():
    for xy in arrs:
        if len(xy) < 2: continue
        t = cKDTree(xy); d, _ = t.query(xy, k=2); nn = d[:, 1]
        r = res[fam]; r["n"] += len(xy)
        for thr in (0.5, 1.0, 2.0, 4.0): r[f"nn_lt_{thr}"] += int((nn < thr).sum())
        for R in (0.5, 1.0, 2.0):  # жадное подавление в случайном порядке силы, 20 повторов
            kept = 0
            for rep in range(20):
                order = rng.permutation(len(xy)); alive = np.ones(len(xy), bool); k = 0
                for i in order:
                    if not alive[i]: continue
                    k += 1
                    for j in t.query_ball_point(xy[i], R): alive[j] = False
                kept += k
            r[f"nms_keep_R{R}"] += kept / 20
for fam, r in res.items():
    n = r["n"]; out[f"finds/{fam}"] = {k: (round(v / n, 4) if k != "n" else int(v)) for k, v in r.items()}
# --- object_poly
for mt in ("maggr", "etgr", "terfl"):
    for fam in sorted({u["family"] for u in U}):
        polys = [(u, g) for u in U if u["cls"] == "object_poly" and u["markup"] == mt and u["family"] == fam for g in u["geoms"] if g.geom_type == "Polygon"]
        if not polys: continue
        A = np.array([g.area for _, g in polys])
        conv = np.array([g.area / g.convex_hull.area for _, g in polys])
        cen_out = np.mean([not g.contains(g.centroid) for _, g in polys])
        # расстояние от внутренней точки GEOS до полюса недоступности, в долях радиуса вписанной окружности
        gaps = []
        byfile = defaultdict(list)
        for u, g in polys: byfile[(u["region"], u["file"])].append(g)
        for gs in byfile.values():
            t = STRtree(gs)
            for i, g in enumerate(gs):
                best = np.inf
                for j in t.query(g.buffer(3.0)):
                    if j != i: best = min(best, g.distance(gs[j]))
                gaps.append(best)
        gaps = np.array(gaps)
        out[f"object_poly/{mt}/{fam}"] = {
            "n": len(polys), "area_m2_p10_p50_p90": [round(float(x), 2) for x in np.percentile(A, [10, 50, 90])],
            "convexity_p10_p50": [round(float(x), 3) for x in np.percentile(conv, [10, 50])],
            "share_convexity_lt_0.8": round(float((conv < 0.8).mean()), 4),
            "share_centroid_outside": round(float(cen_out), 4),
            **{f"share_gap_lt_{d}m": round(float((gaps < d).mean()), 4) for d in (0.1, 0.25, 0.5, 1.0, 2.0)},
        }
# --- находки внутри полигонов object_poly того же участка (MagGr)
for fam in sorted({u["family"] for u in U}):
    inside = tot = 0
    for reg in {u["region"] for u in U if u["family"] == fam}:
        P = [g for u in U if u["region"] == reg and u["cls"] == "object_poly" and u["markup"] == "maggr" for g in u["geoms"]]
        F = [g for u in U if u["region"] == reg and u["cls"] == "finds_points" for g in u["geoms"]]
        if not F: continue
        tot += len(F)
        if P:
            t = STRtree(P)
            inside += sum(any(P[j].contains(p) for j in t.query(p)) for p in F)
    if tot: out[f"finds_inside_maggr_poly/{fam}"] = {"n": tot, "share": round(inside / tot, 4)}
json.dump(out, open("../gt_stats_geophys.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
