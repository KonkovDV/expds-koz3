import sys, time, json, resource, copy
sys.path.insert(0, "/tmp/bot8_half_src")
import numpy as np, torch
torch.set_num_threads(2)
from pathlib import Path
from expds import pipeline, candidates
from expds.lidar import build_dem, hillshade_set, relief_channels
from expds.infer_nn import Segmenter
S = Path("/workspace/koz3_shared/speedup/sample")
sites = [("016", S/"016_Хога/01_Хога_La_точки/01_Хога_La_точки.laz", "EPSG:32632"),
         ("018", S/"018_Курган_Хога/01_Курган_Хога_La_точки/01_Курган_Хога_La_точки.laz", "EPSG:32633")]
t=time.perf_counter(); seg = Segmenter("/tmp/bot8_half_src/models/li_r18_hard2.pt", device="cpu"); t_load=time.perf_counter()-t
cfg = pipeline.load_config(Path("/tmp/bot8_half_src/configs/default.toml"))
cand_p = pipeline._params(candidates.CandidateParams, cfg.get("candidates"))
floors = pipeline.output_floors(cfg, "li")
out = {"model_load_s": t_load, "threads": torch.get_num_threads(), "sites": {}}
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
for name, las, crs in sites:
    t=time.perf_counter(); d = build_dem(las, res=0.1, fill="delaunay", crs=crs); t_dem=time.perf_counter()-t
    r = {"dem_shape": list(d.dem.shape), "dem_s": t_dem}
    for scale, res in (("1x", 0.5), ("0.5x", 1.0)):
        best = {}
        for rep in range(REPS):
            tm = {}
            t=time.perf_counter(); img, tr = relief_channels(d, res, tuple(seg.meta["li_tags"])); tm["channels"]=time.perf_counter()-t
            for tta in (1, 8) if scale=="0.5x" or rep==0 else (1,):
                seg.tta = tta
                t=time.perf_counter(); probs = seg.predict(img); tm[f"predict_tta{tta}"]=time.perf_counter()-t
            seg.tta = 1
            t=time.perf_counter(); c = candidates.detect(d, cand_p); tm["cand_detect"]=time.perf_counter()-t
            kur=["kurgany_povrezhdennye","kurgany_tselye"]; rest=[x for x in seg.classes if x not in kur]
            t=time.perf_counter()
            from expds.infer_nn import probs_to_features
            f1 = probs_to_features(probs[[seg.classes.index(x) for x in rest]], rest, tr, d.crs, tr.a, name, "li", las.name, min_conf=floors)
            kp = probs[[seg.classes.index(k) for k in kur]]
            s = candidates.sample_probs(c, kp, tr)
            sc = np.column_stack([np.zeros(len(s), np.float32), s])
            f2 = candidates.to_features(c, sc, d.crs, name, cand_p, classes=tuple(kur), source=las.name)
            tm["postproc"]=time.perf_counter()-t
            # whole _li_with_model with res overridden
            seg2 = copy.copy(seg); seg2.res = res
            t=time.perf_counter(); ff = pipeline._li_with_model(seg2, d, name, las.name, hillshade_set, cand_p, cfg.get("li"), floors); tm["li_with_model_total"]=time.perf_counter()-t
            tm["n_feats"]=len(ff); tm["img_shape"]=list(img.shape); tm["n_cand"]=len(c)
            for k,v in tm.items():
                if isinstance(v,float): best[k]=min(best.get(k,1e9),v)
                else: best[k]=v
        r[scale]=best
    out["sites"][name]=r
    print(name, json.dumps(r), flush=True)
out["maxrss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/bot8_half_bench/result.json","w"), indent=1)
