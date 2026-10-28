"""Выбор цели (shrink, min_dist, радиус) без своей семьи (leave-family-out): лучший вариант по остальным семьям применяется к семье."""
import json, sys, io, contextlib, math
from collections import defaultdict, Counter
from shapely.ops import unary_union
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials"); sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import compute_metrics as cm
from expds.naming import site_family
def shrink_polygon(g, shrink):
    if shrink <= 0 or g.area <= 0: return g
    r = math.sqrt(g.area / math.pi); s = g.buffer(-shrink * r)
    if s.is_empty or s.area < 0.05 * g.area: s = g.representative_point().buffer(max(r * (1 - shrink), 0.5))
    return s
gt = json.load(open("gt_arch_3857.geojson"))
by = defaultdict(list)
for f in gt["features"]:
    g = cm._extract_geometry(f)
    if g is None: continue
    p = f["properties"]; by[(p["region_name"], p["markup_type"])].append(cm._to_utm(g, "auto"))
calc = cm.F1Calculator()
cfgs = [(s, md, r) for s in (0.0, 0.25, 0.4, 0.6) for md in (4.0, 2.0) for r in (2.0, 0.05)]
tab = defaultdict(Counter)  # fam -> cfg -> tp ; gt
ngt = Counter()
for (reg, mt), gs in by.items():
    fam = site_family(reg); ngt[fam] += len(gs)
    for cfg in cfgs:
        s, md, rad = cfg
        tg = [shrink_polygon(g, s) for g in gs]
        u = unary_union([t.buffer(0.25) for t in tg]); comps = list(getattr(u, "geoms", [u]))
        peaks = sorted(((unary_union([t for t in tg if t.intersects(c)]).centroid, c.area) for c in comps), key=lambda t: -t[1])
        kept = []
        for cen, _ in peaks:
            if all(cen.distance(k) >= md for k in kept): kept.append(cen)
        used = set(); tp = 0
        for cen in kept:
            circ = cen.buffer(rad, quad_segs=16); best, bs = -1, -1
            for j, g in enumerate(gs):
                if j in used: continue
                h, sc = calc._match_polygon_gt(circ, g)
                if h and sc > bs: best, bs = j, sc
            if best >= 0: used.add(best); tp += 1
        tab[fam][cfg] += tp
fams = sorted(ngt)
out = {"per_family_gt": dict(ngt), "folds": {}}
tot_lofo = 0; tot_base = 0
for fam in fams:
    others = Counter()
    for f2 in fams:
        if f2 != fam:
            for cfg in cfgs: others[cfg] += tab[f2][cfg]
    pick = max(cfgs, key=lambda c: (others[c], -c[0]))
    tot_lofo += tab[fam][pick]; tot_base += tab[fam][(0.0, 4.0, 2.0)]
    out["folds"][fam] = dict(pick=pick, tp_pick=tab[fam][pick], tp_current=tab[fam][(0.0, 4.0, 2.0)], gt=ngt[fam])
out["total"] = dict(gt=sum(ngt.values()), tp_current=tot_base, tp_lofo=tot_lofo)
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open("../analysis/arch_target_lofo.json", "w"), ensure_ascii=False, indent=1)
