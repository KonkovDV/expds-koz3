"""Симуляция по эталону: сколько object_poly теряется на склейке аномалий и что дают
ядро polylabel, Multi-Pole Peeling (expds/post.py) и разрезание водоразделом (патч bot-3).

ЭТО НЕ ПРОГОН ДЕТЕКТОРА: растров геофизики в /workspace/data/train-expds-3 нет.
Поле = эталонная маска, размытая гауссом sigma (сценарий растекания аномалии), + шум;
порог 0.5 → связные области (4-связность, как polys_from). Счёт — официальный F1Calculator.
"""
import sys, json, math, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
from scipy import ndimage as ndi
from skimage.draw import polygon as draw_poly
from skimage.measure import find_contours
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from shapely.geometry import Polygon, Point, mapping
from shapely.ops import unary_union, polylabel
from shapely import transform as shp_transform
from pyproj import Transformer
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo/materials")
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
from compute_metrics import F1Calculator
from expds import post
from load_gt import load

AREA = {"maggr": (5.0, 3000.0), "etgr": (0.05, 150.0), "terfl": (0.05, 1000.0)}   # expds/geophys.py AREA_M2
RES = {"maggr": 0.1, "etgr": 0.05, "terfl": 0.05}
R_CIRCLE = 0.05
calc = F1Calculator()
WS_GRID = [("dt", h) for h in (0.1, 0.25, 0.5, 1.0)] + [("field", h) for h in (0.02, 0.05, 0.1, 0.2)]
OUT_TAG = ""

def raster_cluster(polys, res, pad):
    minx, miny, maxx, maxy = unary_union(polys).bounds
    x0, y1 = minx - pad, maxy + pad
    W = int(math.ceil((maxx - minx + 2 * pad) / res)); H = int(math.ceil((maxy - miny + 2 * pad) / res))
    m = np.zeros((H, W), np.float32)
    for p in polys:
        for ring, val in [(p.exterior, 1)] + [(r, 0) for r in p.interiors]:
            xy = np.asarray(ring.coords)
            rr, cc = draw_poly((y1 - xy[:, 1]) / res, (xy[:, 0] - x0) / res, m.shape)
            m[rr, cc] = val
    return m, (x0, y1)

def mask_poly(mask, org, res):
    """Внешний контур бинарной маски одного компонента → Polygon в UTM."""
    cs = find_contours(np.pad(mask.astype(np.float32), 1), 0.5)
    if not cs: return None
    c = max(cs, key=len)
    if len(c) < 4: return None
    x = org[0] + (c[:, 1] - 1 + 0.5) * res; y = org[1] - (c[:, 0] - 1 + 0.5) * res
    g = Polygon(np.c_[x, y]).buffer(0)
    if g.geom_type == "MultiPolygon": g = max(g.geoms, key=lambda q: q.area)
    return g if g.area > 0 else None

def pole_circle_from_mask(mask, org, res):
    dt = ndi.distance_transform_edt(mask)
    r, c = np.unravel_index(np.argmax(dt), dt.shape)
    return Point(org[0] + (c + 0.5) * res, org[1] - (r + 0.5) * res).buffer(R_CIRCLE, quad_segs=8)

def split_ws(comp, field, signal, h, res):
    """Разрезание компонента водоразделом. signal='dt' (перешейки формы) или 'field' (седловины поля)."""
    if signal == "dt":
        s = ndi.distance_transform_edt(comp) * res; hh = h
    else:
        s = np.where(comp, field, 0); hh = h
    mk = h_maxima(s, hh) if s.max() > hh else np.zeros_like(comp, np.uint8)
    mk = mk & comp
    lab, n = ndi.label(mk)
    if n <= 1: return [comp]
    seg = watershed(-s, lab, mask=comp)
    return [seg == k for k in range(1, n + 1)]

def predict_cluster(polys, mt, sigma, noise, rng, ws_grid):
    res = RES[mt]; pad = 3 * sigma + 1.0
    m, org = raster_cluster(polys, res, pad)
    f = ndi.gaussian_filter(m, sigma / res) if sigma > 0 else m
    if noise > 0:
        nz = ndi.gaussian_filter(rng.standard_normal(m.shape).astype(np.float32), max(sigma, 0.25) / res)
        nz *= noise / (nz.std() + 1e-9); f = f + nz
    lab, n = ndi.label(f >= 0.5)
    amin, amax = AREA[mt]
    out = defaultdict(list)
    sl = ndi.find_objects(lab)
    for k, s in enumerate(sl, 1):
        if s is None: continue
        s = tuple(slice(max(0, q.start - 2), q.stop + 2) for q in s)
        comp = lab[s] == k; a = comp.sum() * res * res
        if a < amin or a > amax: continue
        o = (org[0] + s[1].start * res, org[1] - s[0].start * res)
        conf = float(f[s][comp].max())
        g = mask_poly(comp, o, res)
        if g is None: continue
        out["baseline"].append((g, conf))
        out["core_polylabel"].append((polylabel(g, tolerance=0.01).buffer(R_CIRCLE, quad_segs=8), conf))
        out["core_dtpole"].append((pole_circle_from_mask(comp, o, res), conf))
        for signal, h in ws_grid:
            parts = split_ws(comp, f[s], signal, h, res)
            parts = [p for p in parts if p.sum() * res * res >= min(amin, a)] or [comp]
            for p in parts:
                out[f"ws_{signal}_{h}_pole"].append((pole_circle_from_mask(p, o, res), conf))
                gp = mask_poly(p, o, res)
                if gp is not None: out[f"ws_{signal}_{h}_poly"].append((gp, conf))
    return out

def to_fc(items, cls, mt, region, tr):
    feats = []
    for g, conf in items:
        feats.append({"type": "Feature", "geometry": mapping(shp_transform(g, tr, interleaved=False) if False else g_tr(g, tr)),
                      "properties": {"class_name": cls, "confidence": conf, "markup_type": mt, "region_name": region}})
    return {"type": "FeatureCollection", "features": feats}

def g_tr(g, tr):
    return shp_transform(g, lambda xy: np.c_[tr(xy[:, 0], xy[:, 1])])

def score(pred_fc, gt_fc):
    r = calc.calculate_f1(pred_fc, gt_fc)["per_class"]["object_poly"] if "per_class" in calc.calculate_f1(pred_fc, gt_fc) else None
    return r

def main(sigmas=(0.25, 0.5, 1.0), noises=(0.0, 0.15)):
    U = [u for u in load() if u["cls"] == "object_poly" and not ("MagGr_цвет_2" in u["file"] and "Терраса_3" in u["file"])]
    ws_grid = WS_GRID
    rows = []
    for sigma, noise in itertools.product(sigmas, noises):
        rng = np.random.default_rng(12345)
        per = defaultdict(lambda: defaultdict(lambda: {"pred": [], "gt": []}))   # (mt) -> region -> lists
        for u in U:
            mt = u["markup"]; tr = Transformer.from_crs(u["crs"], "EPSG:3857", always_xy=True).transform
            polys = [g for g in u["geoms"] if g.geom_type == "Polygon" and g.area > 0]
            if not polys: continue
            gt_feats = [{"type": "Feature", "geometry": mapping(g_tr(g, tr)), "properties": {"class_name": "object_poly", "markup_type": mt}} for g in polys]
            # кластеры: эталон с буфером 3σ+1 м
            cl = unary_union([p.buffer(3 * sigma + 1.0) for p in polys]); cl = list(getattr(cl, "geoms", [cl]))
            preds = defaultdict(list)
            for c in cl:
                inside = [p for p in polys if c.intersects(p)]
                for k, v in predict_cluster(inside, mt, sigma, noise, rng, ws_grid).items(): preds[k] += v
            bucket = per[mt][u["region"]]
            bucket["gt"] += gt_feats
            # Multi-Pole Peeling из expds/post.py (на полигонах baseline, в EPSG:3857)
            base3857 = [{"geom": g_tr(g, tr), "class_name": "object_poly", "confidence": cf, "markup_type": mt,
                         "region_name": u["region"], "threshold_group": mt} for g, cf in preds["baseline"]]
            mp = post.multipole_object_poly(base3857, {"drop_cluttered_terfl_above": 0})
            bucket["pred"].append((preds, tr, mp))
        for mt, regs in per.items():
            for reg, b in regs.items():
                gt_fc = {"type": "FeatureCollection", "features": b["gt"]}
                methods = set().union(*[set(p.keys()) for p, _, _ in b["pred"]]) | {"multipole_post"}
                fam = next(u["family"] for u in U if u["region"] == reg)
                for meth in sorted(methods):
                    feats = []
                    for preds, tr, mp in b["pred"]:
                        if meth == "multipole_post":
                            feats += [{"type": "Feature", "geometry": mapping(f["geom"]), "properties": {"class_name": "object_poly", "confidence": f["confidence"]}} for f in mp]
                        else:
                            feats += [{"type": "Feature", "geometry": mapping(g_tr(g, tr)), "properties": {"class_name": "object_poly", "confidence": cf}} for g, cf in preds.get(meth, [])]
                    r = calc.calculate_f1({"type": "FeatureCollection", "features": feats}, gt_fc)["per_class"]["object_poly"]
                    rows.append(dict(sigma=sigma, noise=noise, markup=mt, region=reg, family=fam, method=meth,
                                     tp=r["tp"], fp=r["fp"], fn=r["fn"]))
        print("done", sigma, noise, len(rows), flush=True)
    json.dump(rows, open(f"/workspace/koz3_shared/bot-3/sim_poly_rows{OUT_TAG}_s{sigmas[0]}_n{noises[0]}.json" if len(sigmas)==1 and len(noises)==1 else "/workspace/koz3_shared/bot-3/sim_poly_rows.json", "w"), ensure_ascii=False)

if __name__ == "__main__":
    main()
