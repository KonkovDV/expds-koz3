"""Точность бэкендов на картах P (bot-3). python exact.py <участок> <out.jsonl>
Одни признаки → predict lightgbm / lleaves / treelite; max|ΔP| и совпадение наборов кандидатов при t_pix 0.3/0.5/0.7
(poly) и finds (P≥0.2, NMS). 1 поток."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
import numpy as np, tomllib
from expds import geo_pixgbm as G, geophys
from expds.catalog import scan_root
from expds.pipeline import region_crs
cfg = tomllib.load(open("/workspace/koz3_shared/bot-3/repo/configs/default.toml", "rb"))
reg = [r for r in scan_root("/workspace/data/train-expds-3") if r.name.startswith(sys.argv[1])][0]
md = "/workspace/koz3_shared/bot-3/pixgbm/models/all"
B = {b: G.load_models(md, b) for b in ("lightgbm", "lleaves", "treelite")}
for b, m in B.items():
    assert all(x.backend == b for x in m.values()), (b, {k: x.backend for k, x in m.items()})
crs_region = region_crs(reg, cfg); geo_cfg = cfg["geophys"]
def key(fs): return sorted((f["class_name"], round(f["confidence"], 12), f["geom"].wkb) for f in fs)
rows = []
for mt in ("maggr", "etgr", "terfl"):
    files = G.pick_rasters(reg.rasters.get(mt, []))
    if not files: continue
    boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
    noise = geophys.load_mask(reg.noise, mt, crs_region)
    for f in files:
        d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise)
        if not d["crs"]: continue
        P = {}
        for b, m in B.items():
            use = {k: v for k, v in m.items() if k == "poly" or mt == "maggr"}
            P[b] = G.prob_map(d, mt, use, threads=1, procs=1)
        ref = P["lightgbm"]
        for b in ("lleaves", "treelite"):
            r = dict(region=reg.name, mt=mt, raster=f.name, backend=b)
            for k in ref:
                r["maxabs_" + k] = float(np.abs(P[b][k].astype(np.float64) - ref[k]).max())
                if k == "poly":
                    for t in (0.3, 0.5, 0.7):
                        r[f"n_ref_t{t}"] = len(G.poly_candidates(ref[k], d, reg.name, mt, f.name, t_pix=t))
                        r[f"same_t{t}"] = key(G.poly_candidates(ref[k], d, reg.name, mt, f.name, t_pix=t)) == key(G.poly_candidates(P[b][k], d, reg.name, mt, f.name, t_pix=t))
                        r[f"pix_flips_t{t}"] = int(((ref[k] >= t) != (P[b][k] >= t)).sum())
                else:
                    r["n_ref_finds"] = len(G.finds_candidates(ref[k], d, reg.name, mt, f.name))
                    r["same_finds"] = key(G.finds_candidates(ref[k], d, reg.name, mt, f.name)) == key(G.finds_candidates(P[b][k], d, reg.name, mt, f.name))
                    r["pix_flips_finds0.2"] = int(((ref[k] >= 0.2) != (P[b][k] >= 0.2)).sum())
            rows.append(r)
        del d, P
with open(sys.argv[2], "a") as fh:
    for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows", sys.argv[1])
