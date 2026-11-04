"""Диагностика li_half на Эрентуне (НЕ вне обучения: li_r18_hard2 видел Эрентуну).

Мозаика DEM 0,1→0,5 м (как pipeline для крупных LAS), основной проход li на мозаике,
второй проход li_half 1 м + детектор relief 30 м. Полнота GT-курганов ⌀ ≥ 12 м.
"""
import json, os, resource, sys, time, logging
from pathlib import Path
import numpy as np
WT = Path(sys.argv[1]); sys.path.insert(0, str(WT))
OUT = Path(sys.argv[2])
TILE_M = float(os.environ.get("DIAG_TILE_M", "200"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
from shapely.geometry import shape, Point
from shapely.ops import transform as stransform
from pyproj import Transformer
from expds import pipeline, li_half, candidates
from expds.lidar import build_dem_mosaic

REG = Path("/workspace/data/train-expds-3/011_Эрентуна")
LAS = REG / "01_Эрентуна_La_точки/01_Эрентуна_La_точки.laz"
CRS = "EPSG:32633"
KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
def rss(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

cfg = pipeline.load_config(WT / "configs/default.toml")
root = WT
segs = pipeline._segmenters({"li": cfg["models"]["li"]}, "li", root, int(cfg.get("tta", {}).get("li", 1)))
seg = segs[0]
cand_p = pipeline._params(candidates.CandidateParams, cfg.get("candidates"))
hp = li_half.params(cfg)
t0 = time.monotonic()
MC = OUT / "mosaic_cache.npz"
if MC.exists():
    from expds.lidar import Dem
    z = np.load(MC, allow_pickle=False)
    mosaic = Dem(dem=z["dem"], left=float(z["left"]), top=float(z["top"]), res=float(z["res"]), crs=CRS,
                 info=json.loads(str(z["info"])))
else:
    mosaic = build_dem_mosaic(LAS, res=0.1, work_res=0.5, tile_m=TILE_M, halo_m=40.0, crs=CRS)
    np.savez_compressed(MC, dem=mosaic.dem, left=mosaic.left, top=mosaic.top, res=mosaic.res,
                        info=json.dumps(mosaic.info, default=str))
t_mos = time.monotonic() - t0
print("mosaic", mosaic.dem.shape, mosaic.res, "s", round(t_mos, 1), "rss", rss(), flush=True)
t0 = time.monotonic()
main = [f for f in pipeline._li_with_model(seg, mosaic, REG.name, LAS.name, pipeline.hillshade_set if hasattr(pipeline, "hillshade_set") else __import__("expds.lidar", fromlist=["x"]).hillshade_set,
                                           cand_p, cfg.get("li"), None) if f["class_name"] in KUR]
t_main = time.monotonic() - t0
print("main", len(main), "s", round(t_main, 1), "rss", rss(), flush=True)
t0 = time.monotonic()
dump = OUT / "dump"; dump.mkdir(parents=True, exist_ok=True)
half = li_half.half_pass(seg, mosaic, REG.name, LAS.name, cand_p, hp, dump, {"mosaic": True, "diag": True})
t_half = time.monotonic() - t0
print("half", len(half), "s", round(t_half, 1), "rss", rss(), flush=True)
print("feat keys", sorted(main[0].keys()) if main else None, flush=True)

_TR = Transformer.from_crs("EPSG:3857", CRS, always_xy=True)
def geom(f):
    g = f["geom"]            # pipeline-объекты: geom в EPSG:3857 (writer), original_crs — UTM участка
    return stransform(_TR.transform, g) if abs(g.centroid.x) > 1_000_000 else g
def conf(f): return float(f.get("confidence", f.get("score", 0.0)))

# GT
gt = []
to_utm = Transformer.from_crs("EPSG:3857", CRS, always_xy=True)
for cls, fn in (("kurgany_povrezhdennye", "поврежденные"), ("kurgany_tselye", "целые")):
    p = REG / f"06_Эрентуна_разметка/Эрентуна_Li_курганы_{fn}.geojson"
    for ft in json.load(open(p))["features"]:
        g = stransform(to_utm.transform, shape(ft["geometry"]))
        gt.append({"cls": cls, "g": g, "d": 2 * np.sqrt(g.area / np.pi)})
ds = np.array([x["d"] for x in gt])
big = [x for x in gt if x["d"] >= 12.0]
print("GT", len(gt), "⌀ median", round(float(np.median(ds)), 1), "p90", round(float(np.percentile(ds, 90)), 1), "≥12:", len(big), flush=True)

def centers(fs): return [(geom(f).centroid, f["class_name"], conf(f)) for f in fs]
def recall(cs, gts):
    hit_any = sum(any(x["g"].contains(c) for c, _, _ in cs) for x in gts)
    hit_cls = sum(any(k == x["cls"] and x["g"].contains(c) for c, k, _ in cs) for x in gts)
    return hit_any, hit_cls

thr_m = cfg["thresholds_by_markup"]["li_mosaic"]
main_kept = [f for f in main if conf(f) >= thr_m[f["class_name"]]]
new_all = li_half.new_only(half, main_kept, hp)
res = {"label": "ДИАГНОСТИКА, не вне обучения (li_r18_hard2 обучался в т.ч. на Эрентуне)",
       "data": {"las": str(LAS), "gt": "train-expds-3/011_Эрентуна/06_*/Эрентуна_Li_курганы_{поврежденные,целые}.geojson",
                "weights": "models/li_r18_hard2.pt (repo main)"},
       "mosaic": {"shape": list(mosaic.dem.shape), "res": mosaic.res, "tile_m": TILE_M, "halo_m": 40.0},
       "seconds": {"mosaic": round(t_mos, 1), "main": round(t_main, 1), "half": round(t_half, 1)},
       "gt": {"n": len(gt), "d_median": round(float(np.median(ds)), 2), "d_p90": round(float(np.percentile(ds, 90)), 2),
              "n_ge12": len(big), "n_ge12_by_cls": {k: sum(x["cls"] == k for x in big) for k in KUR}},
       "li_mosaic_thr": thr_m, "rows": []}
def row(name, fs):
    cs = centers(fs); a, c = recall(cs, big); a_all, c_all = recall(cs, gt)
    hits_c = sum(any(x["g"].contains(c) for x in gt) for c, _, _ in cs)
    r = {"name": name, "n_circles": len(fs), "circles_in_any_gt": hits_c, "recall_ge12_any": f"{a}/{len(big)}", "recall_ge12_cls": f"{c}/{len(big)}",
         "recall_all_any": f"{a_all}/{len(gt)}"}
    res["rows"].append(r); print(r, flush=True)
row("main_all_candidates", main)
row("main_kept@li_mosaic", main_kept)
row("half_all_candidates", half)
row("half_new_all(12m)", new_all)
row("main_kept+half_new_all", main_kept + new_all)
for t in (0.3, 0.5, 0.7, 0.9, 0.95):
    nt = [f for f in new_all if conf(f) >= t]
    row(f"main_kept+half_new@{t}", main_kept + nt)
# контроль: тот же основной проход с более низким порогом — столько же кругов, сколько main_kept+half_new@t
rest = sorted([f for f in main if f not in main_kept], key=conf, reverse=True)
for t in (0.3, 0.5, 0.9):
    n_extra = sum(conf(f) >= t for f in new_all)
    row(f"control_main_kept+top{n_extra}_main_below_thr(=n of half@{t})", main_kept + rest[:n_extra])
res["maxrss_mb"] = rss()
(OUT / "eruntuna_diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("maxrss", rss())
