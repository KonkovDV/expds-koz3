"""Тест малого бюджета PIXGBM (bot-3): geophys_feats (как в pipeline.run) → watchdog.finish → write_geojson → validate.
python tiny.py <case> <budget|none> <model_dir> <out_prefix>; case: classic | pix. Процесс на случай (лестница липкая)."""
import os, sys, json, time, tomllib
sys.path.insert(0, "/workspace/koz3_shared/bot-3/repo")
from expds import geo_pixgbm as G, writer, watchdog
from expds.catalog import scan_root
from expds.pipeline import TimeGuard, geophys_feats
case, budget, md, pref = sys.argv[1:5]
cfg = tomllib.load(open("/workspace/koz3_shared/bot-3/repo/configs/default.toml", "rb"))
thr = tomllib.load(open("/workspace/koz3_shared/bot-3/pixgbm/thresholds_v2.toml", "rb"))["thresholds_by_markup"]
cfg.setdefault("thresholds_by_markup", {}).update(thr)
px = cfg["geophys"]["pixgbm"]
px.update(enabled=(case == "pix"), replace=True, model_dir=md, backend="lleaves")
if budget != "none": px["budget_s"] = float(budget)
guard = TimeGuard(1800)
c0 = os.times(); t0 = time.monotonic()
feats = []
for reg in scan_root("/workspace/data/train-expds-3"):
    if reg.name[:3] in tuple(os.environ.get("TINY_REGS", "201 208 216").split()):
        feats += geophys_feats(reg, cfg, guard)
out = watchdog.finish(feats, cfg)
p = writer.write_geojson(out, pref + ".geojson")
c1 = os.times()
rep = writer.validate(p)
fc = json.load(open(p))
res = dict(case=case, budget=budget, model_dir=md, wall=round(time.monotonic() - t0, 2),
           cpu=round(c1.user + c1.system + c1.children_user + c1.children_system - c0.user - c0.system - c0.children_user - c0.children_system, 2),
           since_proc_start=round(time.monotonic() - G.process_t0(), 2),
           ok=rep["ok"], errors=rep["errors"][:5], warnings=rep["warnings"][:3], summary=rep["summary"],
           crs=sorted({f["properties"].get("crs") for f in fc["features"]}), geom_types=sorted({f["geometry"]["type"] for f in fc["features"]}),
           steps=G.PHASE["steps"], level=G.PHASE["level"],
           by_group={})
for f in fc["features"]:
    g = f["properties"].get("markup_type", "?") + "|" + f["properties"]["class_name"]
    res["by_group"][g] = res["by_group"].get(g, 0) + 1
print(json.dumps(res, ensure_ascii=False))
open(pref + ".json", "w").write(json.dumps(res, ensure_ascii=False, indent=1))
