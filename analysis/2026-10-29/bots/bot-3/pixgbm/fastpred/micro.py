"""Микро-бенч predict: X из окон участка 204 (maggr) → lightgbm / lleaves / tl2cgen. python micro.py <nthreads>"""
import os, sys, time, json
nt = int(sys.argv[1]); os.environ["OMP_NUM_THREADS"] = str(nt)
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
import numpy as np, tomllib, lightgbm as lgb
from expds import geo_pixgbm as G, geophys
from expds.catalog import scan_root
from expds.pipeline import region_crs
MD = "/workspace/koz3_shared/bot-3/pixgbm/models/all"; CACHE = "/workspace/koz3_shared/bot-3/pixgbm/fastpred/cache"
os.makedirs(CACHE, exist_ok=True)
cfg = tomllib.load(open("/workspace/koz3_shared/bot-3/repo/configs/default.toml", "rb"))
reg = [r for r in scan_root("/workspace/data/train-expds-3") if r.name.startswith("204")][0]
crs = region_crs(reg, cfg); f = G.pick_rasters(reg.rasters["maggr"])[0]
d = G.load_map(f, 0.1, crs, geophys.survey_boundary(reg.boundaries, "maggr", crs, cfg.get("geophys", {})), geophys.load_mask(reg.noise, "maggr", crs))
(c5, c15), b = G.trends(d["z"], d["valid"], d["gsd"]); H, W = d["z"].shape; use = d["valid"] & ~d["noise"]
bst = lgb.Booster(model_file=f"{MD}/poly.txt"); names = bst.feature_name(); Xs = []
for ty, tx, (y0, x0, y1, x1) in G.windows(H, W, d["gsd"]):
    rr, cc = np.nonzero(use[ty:min(ty+G.CORE,H), tx:min(tx+G.CORE,W)])
    if not rr.size: continue
    rr, cc = rr + ty - y0, cc + tx - x0; h, w = y1 - y0, x1 - x0
    F = G.features_at(d["z"][y0:y1, x0:x1], d["zd"][y0:y1, x0:x1], d["valid"][y0:y1, x0:x1], G._up(c5, b, y0, x0, h, w), G._up(c15, b, y0, x0, h, w), d["gsd"], 0, rr, cc)
    Xs.append(np.column_stack([F[n] for n in names]))
    if sum(len(x) for x in Xs) > 200000: break
X = np.vstack(Xs); del d, Xs
print("X", X.shape, X.dtype, "nan", int(np.isnan(X).sum()), flush=True)
res = dict(nt=nt, n=len(X))
def tm(fn, rep=2):
    best = 1e9
    for _ in range(rep):
        t = time.perf_counter(); p = fn(); best = min(best, time.perf_counter() - t)
    return best, p
out = {}
for k in ("poly", "finds"):
    b = lgb.Booster(model_file=f"{MD}/{k}.txt")
    t, p0 = tm(lambda: b.predict(X, num_threads=nt)); out[k] = {"lightgbm": dict(s=t)}; p032 = p0.astype(np.float32)
    import lleaves
    for tag, cpu, feats in (("lleaves_native", None, None), ("lleaves_x86-64-v3", "x86-64-v3", ""), ("lleaves_x86-64", "x86-64", "")):
        m = lleaves.Model(model_file=f"{MD}/{k}.txt"); cp = f"{CACHE}/{k}_{tag}.o"; fresh = not os.path.exists(cp)
        t1 = time.perf_counter(); m.compile(cache=cp, target_cpu=cpu, target_cpu_features=feats); tc = time.perf_counter() - t1
        X64 = X.astype(np.float64)
        t, p = tm(lambda: m.predict(X64, n_jobs=nt))
        out[k][tag] = dict(s=t, compile_s=tc, compiled_fresh=fresh, obj_bytes=os.path.getsize(cp), maxabs=float(np.abs(p - p0).max()),
                           maxabs_f32=float(np.abs(p.astype(np.float32) - p032).max()))
        print(k, tag, out[k][tag], flush=True)
    import treelite, tl2cgen
    tlm = treelite.frontend.load_lightgbm_model(f"{MD}/{k}.txt"); lib = f"{CACHE}/{k}_tl2cgen.so"; fresh = not os.path.exists(lib)
    t1 = time.perf_counter()
    if fresh: tl2cgen.export_lib(tlm, toolchain="gcc", libpath=lib, params={"parallel_comp": 8}, nthread=nt, options=["-O2"])
    tc = time.perf_counter() - t1
    pr = tl2cgen.Predictor(lib, nthread=nt); dm = tl2cgen.DMatrix(X.astype(np.float64))
    t, p = tm(lambda: pr.predict(dm)); p = np.asarray(p).reshape(-1)
    out[k]["tl2cgen"] = dict(s=t, compile_s=tc, compiled_fresh=fresh, obj_bytes=os.path.getsize(lib), maxabs=float(np.abs(p - p0).max()), maxabs_f32=float(np.abs(p.astype(np.float32) - p032).max()))
    print(k, "tl2cgen", out[k]["tl2cgen"], flush=True)
    for kk, v in out[k].items(): v["speedup"] = round(out[k]["lightgbm"]["s"] / v["s"], 2)
res["out"] = out
import resource; res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
open("micro.jsonl", "a").write(json.dumps(res) + "\n"); print(json.dumps(res, indent=1))
