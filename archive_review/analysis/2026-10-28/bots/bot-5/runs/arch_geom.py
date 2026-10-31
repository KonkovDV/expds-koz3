"""Форма эталона архитектуры и геометрический оракул официальной метрикой (compute_metrics.py)."""
import json, sys, math, random, io, contextlib
from collections import defaultdict
import numpy as np
from shapely.geometry import shape, mapping, Point
from shapely.ops import unary_union, transform as stransform
from shapely.ops import polylabel
from pyproj import Transformer
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials")
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import compute_metrics as cm
from expds.naming import site_family

gt = json.load(open("gt_arch_3857.geojson"))
recs = []
for f in gt["features"]:
    g3857 = cm._extract_geometry(f)
    if g3857 is None:
        continue
    gu = cm._to_utm(g3857, "auto")
    p = f["properties"]
    recs.append(dict(f=f, g3857=g3857, gu=gu, region=p["region_name"], mt=p["markup_type"],
                     fam=site_family(p["region_name"])))
print("GT в метрике:", len(recs))

def utm_tr(g3857):
    c = g3857.centroid
    lon, lat = cm._get_transformer_3857_to_wgs84().transform(c.x, c.y)
    epsg = cm._to_utm_zone_epsg(lon, lat)
    return Transformer.from_crs(epsg, 3857, always_xy=True).transform

# ---- форма
rows = []
for r in recs:
    g = r["gu"]
    A, P = g.area, g.length
    mrr = g.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    e = sorted([math.dist((xs[0], ys[0]), (xs[1], ys[1])), math.dist((xs[1], ys[1]), (xs[2], ys[2]))])
    pl = polylabel(g, tolerance=0.01)
    rin = g.exterior.distance(pl) if not g.interiors else g.boundary.distance(pl)
    rows.append(dict(region=r["region"], mt=r["mt"], fam=r["fam"], area=A, perim=P,
                     compact=4 * math.pi * A / P ** 2 if P else 0, mrr_w=e[0], mrr_l=e[1],
                     elong=e[1] / e[0] if e[0] > 0 else float("inf"), r_in=rin, holes=len(g.interiors),
                     centroid_in=g.contains(g.centroid), fill_mrr=A / mrr.area if mrr.area else 0))
    r["shape"] = rows[-1]
json.dump(rows, open("../analysis/arch_shape_rows.json", "w"), ensure_ascii=False, indent=0)

def q(v, ps=(10, 50, 90)):
    return [round(float(np.percentile(v, p)), 2) for p in ps]

summ = {}
for mt in ["sp_or", "or", "ae", "li", "ALL"]:
    rr = [x for x in rows if mt == "ALL" or x["mt"] == mt]
    if not rr:
        continue
    a = np.array([x["area"] for x in rr])
    summ[mt] = dict(n=len(rr), regions=len({x["region"] for x in rr}), families=len({x["fam"] for x in rr}),
                    area_p10_50_90=q(a), r_in_p10_50_90=q([x["r_in"] for x in rr]),
                    width_mrr_p10_50_90=q([x["mrr_w"] for x in rr]), elong_p10_50_90=q([x["elong"] for x in rr]),
                    compact_p10_50_90=q([x["compact"] for x in rr]), fill_mrr_p50=q([x["fill_mrr"] for x in rr], (50,)),
                    share_area_lt_12_57=round(float((a < math.pi * 4).mean()), 3),
                    share_area_lt_6_28=round(float((a < math.pi * 2).mean()), 3),
                    share_r_in_lt_1=round(float(np.mean([x["r_in"] < 1.0 for x in rr])), 3),
                    share_centroid_outside=round(float(np.mean([not x["centroid_in"] for x in rr])), 3),
                    share_with_holes=round(float(np.mean([x["holes"] > 0 for x in rr])), 3))
# соседи ближе 4 м (min_dist_m сети) и касающиеся на сетке 0,5 м
cent = defaultdict(list)
for r in recs:
    cent[r["region"]].append(r)
close4 = 0; touching = 0
for reg, lst in cent.items():
    for i, a in enumerate(lst):
        ca = a["gu"].centroid
        if any(j != i and ca.distance(b["gu"].centroid) < 4.0 for j, b in enumerate(lst)):
            close4 += 1
        if any(j != i and a["gu"].distance(b["gu"]) < 0.5 for j, b in enumerate(lst)):
            touching += 1
        a["touch"] = any(j != i and a["gu"].distance(b["gu"]) < 0.5 for j, b in enumerate(lst))
summ["neighbours"] = dict(centroid_closer_4m=close4, gap_lt_0_5m=touching, n=len(recs))

# ---- оракул
def circ(pt_utm, r, tr):
    c = Point(pt_utm).buffer(r, quad_segs=16)
    return stransform(tr, c)

def run(preds, sub):
    fc_p = {"type": "FeatureCollection", "features": preds}
    fc_g = {"type": "FeatureCollection", "features": [r["f"] for r in sub]}
    with contextlib.redirect_stdout(io.StringIO()):
        res = cm.F1Calculator().calculate_f1(fc_p, fc_g)
    pc = res["per_class"]["arkhitektury"] if "per_class" in res else None
    return pc

def centers(r, mode, jitter, rng):
    g = r["gu"]
    if mode == "rep":
        c = g.representative_point()
    elif mode == "polylabel":
        c = polylabel(g, tolerance=0.01)
    else:
        c = g.centroid
    x, y = c.x, c.y
    if jitter > 0:
        t = rng.uniform(0, 2 * math.pi)
        x += jitter * math.cos(t); y += jitter * math.sin(t)
    return (x, y)

RADII = [2.0, 1.0, 0.5, 0.25, 0.05]
results = []
for mode in ["rep", "centroid", "polylabel"]:
    for jitter in ([0.0, 0.5, 1.0, 1.5] if mode == "centroid" else [0.0]):
        seeds = [0, 1, 2] if jitter > 0 else [0]
        for rad in RADII:
            agg = defaultdict(lambda: [0, 0])
            for s in seeds:
                rng = random.Random(s)
                hits = []
                for r in recs:
                    tr = utm_tr(r["g3857"])
                    c = circ(centers(r, mode, jitter, rng), rad, tr)
                    # отдельный прогон на объект: только геометрия попадания, без чужих эталонов
                    pc_g, pc_c = r["gu"], stransform(lambda x, y, z=None: (x, y), Point(0, 0))
                    pu = cm._to_utm(c, "auto")
                    hit, _ = cm.F1Calculator()._match_polygon_gt(pu, r["gu"])
                    r.setdefault("hits", {})[(mode, jitter, rad, s)] = hit
                    for key in (r["mt"], "ALL", "fam:" + r["fam"]):
                        agg[key][0] += int(hit); agg[key][1] += 1
            results.append(dict(mode=mode, jitter=jitter, radius=rad,
                                recall={k: round(v[0] / v[1], 4) for k, v in agg.items()},
                                n={k: v[1] // len(seeds) for k, v in agg.items()}))
# сверка: полный прогон F1Calculator на всех эталонах (пулинг, жадный матчинг) для rep и centroid без сдвига
check = {}
for mode in ["rep", "centroid"]:
    for rad in RADII:
        rng = random.Random(0)
        preds = []
        for r in recs:
            c = circ(centers(r, mode, 0.0, rng), rad, utm_tr(r["g3857"]))
            preds.append({"type": "Feature", "geometry": mapping(c),
                          "properties": {"class_name": "arkhitektury", "confidence": 1.0,
                                         "region_name": r["region"], "markup_type": r["mt"]}})
        pc = run(preds, recs)
        check[f"{mode}_r{rad}"] = {k: pc[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1") if k in pc}
json.dump(dict(shape=summ, oracle=results, official_full_run=check),
          open("../analysis/arch_geom_oracle.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summ, ensure_ascii=False, indent=1))
for x in results:
    print(x["mode"], x["jitter"], x["radius"], {k: v for k, v in x["recall"].items() if not k.startswith("fam:")})
print(json.dumps(check, indent=1))
