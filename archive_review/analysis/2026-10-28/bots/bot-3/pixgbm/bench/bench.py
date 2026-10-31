"""Замер времени PIXGBM по этапам на участок (bot-3). python bench.py <участок> <потоки> <out.jsonl>
Этапы: load (чтение+маски), trend (медианный детренд), feat (features_at по окнам), pred (LightGBM poly+finds),
cand (полигоны/находки). wall — time.perf_counter, cpu — time.process_time (видно, сколько отняла конкуренция за ядра)."""
import os, sys, time, json, resource
reg_name, nt, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.environ["OMP_NUM_THREADS"] = str(nt)
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
import numpy as np, tomllib, lightgbm as lgb
from expds import geo_pixgbm as G, geophys
from expds.catalog import scan_root
from expds.pipeline import region_crs
G.LGBM["num_threads"] = nt
cfg = tomllib.load(open("/workspace/koz3_shared/bot-3/repo/configs/default.toml", "rb"))
reg = [r for r in scan_root("/workspace/data/train-expds-3") if r.name.startswith(reg_name)][0]
md = "/workspace/koz3_shared/bot-3/pixgbm/models/all"
t0 = time.perf_counter()
bst = {k: lgb.Booster(model_file=f"{md}/{k}.txt") for k in ("poly", "finds")}
t_model = time.perf_counter() - t0
crs_region = region_crs(reg, cfg); geo_cfg = cfg.get("geophys", {})
rows = []
for mt in ("maggr", "etgr", "terfl"):
    files = G.pick_rasters(reg.rasters.get(mt, []))
    if not files: continue
    boundary = geophys.survey_boundary(reg.boundaries, mt, crs_region, geo_cfg)
    noise = geophys.load_mask(reg.noise, mt, crs_region)
    use = {k: v for k, v in bst.items() if k == "poly" or mt == "maggr"}
    for f in files:
        T = {}; C = {}
        def tic(): return time.perf_counter(), time.process_time()
        def toc(k, s):
            T[k] = T.get(k, 0) + time.perf_counter() - s[0]; C[k] = C.get(k, 0) + time.process_time() - s[1]
        s = tic(); d = G.load_map(f, G.GSD[mt], crs_region, boundary, noise); toc("load", s)
        if not d["crs"]: continue
        s = tic(); (c5, c15), b = G.trends(d["z"], d["valid"], d["gsd"]); toc("trend", s)
        H, W = d["z"].shape; usem = d["valid"] & ~d["noise"]
        P = {k: np.zeros((H, W), np.float32) for k in use}
        for ty, tx, (y0, x0, y1, x1) in G.windows(H, W, d["gsd"]):
            cy1, cx1 = min(ty + G.CORE, H), min(tx + G.CORE, W)
            rr, cc = np.nonzero(usem[ty:cy1, tx:cx1])
            if not rr.size: continue
            rr, cc = rr + ty - y0, cc + tx - x0; h, w = y1 - y0, x1 - x0
            s = tic()
            F = G.features_at(d["z"][y0:y1, x0:x1], d["zd"][y0:y1, x0:x1], d["valid"][y0:y1, x0:x1],
                              G._up(c5, b, y0, x0, h, w), G._up(c15, b, y0, x0, h, w), d["gsd"], G.MT_ID[mt], rr, cc)
            toc("feat", s)
            for k, bs in use.items():
                s = tic()
                X = np.column_stack([F[n] for n in bs.feature_name()])
                P[k][rr + y0, cc + x0] = bs.predict(X, num_threads=nt).astype(np.float32)
                toc("pred_" + k, s)
            del F
        s = tic(); n = 0
        if "poly" in P: n += len(G.poly_candidates(P["poly"], d, reg.name, mt, f.name))
        if "finds" in P: n += len(G.finds_candidates(P["finds"], d, reg.name, mt, f.name))
        toc("cand", s)
        rows.append(dict(region=reg.name, mt=mt, raster=f.name, threads=nt, H=H, W=W, mpx=H * W / 1e6,
                         mpx_valid=float(usem.sum()) / 1e6, wall={k: round(v, 2) for k, v in T.items()},
                         cpu={k: round(v, 2) for k, v in C.items()}, total=round(sum(T.values()), 2), n_cand=n))
        del d, P
tot = dict(region=reg.name, threads=nt, summary=True, model_load=round(t_model, 2),
           total=round(sum(r["total"] for r in rows), 2), mpx=round(sum(r["mpx"] for r in rows), 2),
           mpx_valid=round(sum(r["mpx_valid"] for r in rows), 2), n_rasters=len(rows),
           maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
           load1=os.getloadavg()[0])
with open(out, "a") as fh:
    for r in rows + [tot]: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(tot, ensure_ascii=False))
