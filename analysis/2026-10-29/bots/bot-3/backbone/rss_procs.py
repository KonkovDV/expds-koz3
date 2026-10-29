#!/usr/bin/env python3
"""Пиковый RSS стадии признаков PIXGBM (G.prob_map) при procs=1/2/4 на одном участке: родитель + дети (psutil, опрос 0,2 с).
python backbone/rss_procs.py <участок> <procs>  → JSON строка"""
import json, os, sys, threading, time
from pathlib import Path
import psutil
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo"); sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo/scripts")
import pixgbm as P
from expds import geo_pixgbm as G, geophys
from expds.pipeline import load_config, region_crs
from expds.naming import site_family
import lightgbm as lgb

site, procs = sys.argv[1], int(sys.argv[2])
cfg = load_config("/workspace/koz3_shared/bot-3/repo/configs/default.toml"); gc = cfg.get("geophys", {})
reg = next(P.regions("/workspace/data/train-expds-3", [site]))
md = Path("/workspace/koz3_shared/bot-3/pixgbm/models") / site_family(reg.name)
bst = {k: lgb.Booster(model_file=str(md / f"{k}.txt")) for k in ("poly", "finds")}
crs = region_crs(reg, cfg); mt = "maggr"
f = G.pick_rasters(reg.rasters[mt])[0]
d = G.load_map(f, G.GSD[mt], crs, geophys.survey_boundary(reg.boundaries, mt, crs, gc), geophys.load_mask(reg.noise, mt, crs))
me = psutil.Process()
base = me.memory_info().rss
peak = dict(sum=0, parent=0, children_max=0, n_children=0, children_sum=0)
stop = False
def mon():
    while not stop:
        try:
            ch = me.children(recursive=True); r = [c.memory_info().rss for c in ch]
        except psutil.Error:
            continue
        p = me.memory_info().rss
        peak["parent"] = max(peak["parent"], p); peak["sum"] = max(peak["sum"], p + sum(r))
        if r:
            peak["children_max"] = max(peak["children_max"], max(r)); peak["children_sum"] = max(peak["children_sum"], sum(r))
            peak["n_children"] = max(peak["n_children"], len(r))
        time.sleep(0.2)
th = threading.Thread(target=mon, daemon=True); th.start()
tm = {}; t = time.time()
G.prob_map(d, mt, bst, threads=2, procs=procs, timing=tm)
sec = time.time() - t; stop = True; th.join()
print(json.dumps(dict(site=reg.name, raster=f.name, mpx=round(d["z"].size / 1e6, 2), procs=procs, sec=round(sec, 1),
                      rss_before_mb=round(base / 2**20), **{k: (round(v / 2**20) if k != "n_children" else v) for k, v in peak.items()},
                      timing={k: round(v, 1) for k, v in tm.items()}), ensure_ascii=False), flush=True)
