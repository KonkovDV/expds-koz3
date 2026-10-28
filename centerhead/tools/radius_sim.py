"""Какой радиус круга-выхода максимизирует попадание по официальной метрике (compute_metrics.py koz3-1.5.0).
Эталон: Li-курганы gt_train_v2 (полигоны). Центр = centroid (или representative_point, если centroid снаружи),
затем сдвиг d м в 8 направлениях. Попадание = _match_polygon_gt официального кода (cov>=0.5 И rep-point внутри)."""
import sys, json, math, collections
import numpy as np
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo/materials")
import compute_metrics as CM
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
calc = CM.F1Calculator()
d = json.load(open("/workspace/rerank/gt_train_v2.geojson"))
G = collections.defaultdict(list)
for f in d["features"]:
    p = f["properties"]
    if p.get("markup_type") == "li" and p.get("class_name") in ("kurgany_tselye", "kurgany_povrezhdennye"):
        g = CM._to_utm(shape(f["geometry"]), "auto")
        G[p["class_name"]].append((p.get("region_name"), g))
del d
D = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]; R = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
out = {}
for cls, gl in G.items():
    eqr = np.array([math.sqrt(g.area / math.pi) for _, g in gl])
    hit = np.zeros((len(D), len(R))); n = 0
    for _, g in gl:
        c = g.centroid
        if not g.contains(c): c = g.representative_point()
        n += 1
        for i, dd in enumerate(D):
            for a in range(8):
                ang = 2 * math.pi * a / 8; p = Point(c.x + dd * math.cos(ang), c.y + dd * math.sin(ang))
                for j, r in enumerate(R):
                    circ = CM._point_to_circle(p, r, calc.circle_points)
                    h, _ = calc._match_polygon_gt(circ, g)
                    hit[i, j] += h
    hit /= (n * 8)
    # ближайший сосед того же класса (любой Li-курган) — для NMS
    allg = [g for c2 in G.values() for _, g in c2]
    cent = np.array([[g.centroid.x, g.centroid.y] for g in allg])
    from scipy.spatial import cKDTree
    t = cKDTree(cent); dist, _ = t.query(np.array([[g.centroid.x, g.centroid.y] for _, g in gl]), k=2)
    nn = dist[:, 1]
    out[cls] = dict(n=n, eq_radius_m_q=dict(zip(["p5", "p25", "p50", "p75", "p95"], np.round(np.percentile(eqr, [5, 25, 50, 75, 95]), 2).tolist())),
                    nn_centroid_dist_m_q=dict(zip(["p5", "p25", "p50", "p75", "p95"], np.round(np.percentile(nn, [5, 25, 50, 75, 95]), 2).tolist())),
                    frac_nn_lt_eqr_sum=float(np.mean(nn < 2 * np.median(eqr))),
                    hit=[[round(float(v), 4) for v in row] for row in hit], offsets_m=D, radii_m=R)
    print(cls, json.dumps(out[cls], ensure_ascii=False))
json.dump(out, open("/workspace/koz3_shared/bot-2/work/centerhead/radius_sim.json", "w"), indent=1, ensure_ascii=False)
