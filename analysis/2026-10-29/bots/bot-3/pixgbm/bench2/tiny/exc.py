"""Путь исключения (bot-3): prob_map бросает RuntimeError → каждый растр «error», выход = классика."""
import os, sys, json, tomllib
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
from expds import geo_pixgbm as G, watchdog, writer
from expds.catalog import scan_root
from expds.pipeline import TimeGuard, geophys_feats
cfg = tomllib.load(open("/workspace/koz3_shared/bot-3/repo/configs/default.toml", "rb"))
cfg.setdefault("thresholds_by_markup", {}).update(tomllib.load(open("../../thresholds_v2.toml", "rb"))["thresholds_by_markup"])
cfg["geophys"]["pixgbm"].update(enabled=True, replace=True, model_dir="/workspace/koz3_shared/bot-3/pixgbm/models/all", backend="lleaves", budget_s=1e9)
def boom(*a, **k): raise RuntimeError("тест исключения")
G.prob_map = boom
feats = []
for reg in scan_root("/workspace/data/train-expds-3"):
    if reg.name[:3] in ("201", "208", "216"): feats += geophys_feats(reg, cfg, TimeGuard(1800))
p = writer.write_geojson(watchdog.finish(feats, cfg), "exc.geojson")
same = json.dumps(json.load(open(p)), sort_keys=True) == json.dumps(json.load(open("classic.geojson")), sort_keys=True)
print(json.dumps(dict(steps=G.PHASE["steps"], ok=writer.validate(p)["ok"], same_as_classic=same)))
