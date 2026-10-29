"""Мпикс валидной площади (valid & ~noise) по участку и методу — для s/Мпикс (bot-3). 1 поток."""
import os, sys, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
import tomllib
from expds import geo_pixgbm as G, geophys
from expds.catalog import scan_root
from expds.pipeline import region_crs
cfg = tomllib.load(open("/workspace/koz3_shared/bot-3/repo/configs/default.toml", "rb"))
out = {}
for reg in scan_root("/workspace/data/train-expds-3"):
    if reg.name[:3] not in "201 204 208 212 216 217 200".split(): continue
    crs = region_crs(reg, cfg); gc = cfg["geophys"]
    for mt in ("maggr", "etgr", "terfl"):
        for f in G.pick_rasters(reg.rasters.get(mt, [])):
            d = G.load_map(f, G.GSD[mt], crs, geophys.survey_boundary(reg.boundaries, mt, crs, gc), geophys.load_mask(reg.noise, mt, crs))
            if d["crs"]:
                out.setdefault(reg.name[:3], {}).setdefault(mt, 0.0)
                out[reg.name[:3]][mt] += float((d["valid"] & ~d["noise"]).sum()) / 1e6
            del d
print(json.dumps(out)); json.dump(out, open("mpx.json", "w"), indent=1)
