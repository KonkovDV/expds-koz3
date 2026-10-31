"""Худший случай по числу объектов: выдача finish() всех 103 семей (весь train, ~210 регионов) в ОДИН файл.
Семьи гоняются по одной (память), выдачи склеиваются, затем post.cap_features + writer.write_geojson + validate."""
import json, resource, sys, time, tomllib
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from shapely.geometry import shape
from expds import post, watchdog, writer
cfg = tomllib.loads(Path("configs/default.toml").read_text(encoding="utf-8"))
allk, t0 = [], time.monotonic()
for p in sorted(Path("/workspace/koz3_shared/bot-10/work/fam").glob("*.jsonl")):
    feats = [dict(json.loads(l)["properties"], geom=shape(json.loads(l)["geometry"])) for l in open(p, encoding="utf-8")]
    allk += watchdog.finish(feats, cfg); del feats
t1 = time.monotonic(); capped = post.cap_features(allk)
out = Path(__file__).with_name("tmp_all"); out.mkdir(exist_ok=True)
w = writer.write_geojson(capped, out / "result.geojson"); t2 = time.monotonic()
v = writer.validate(w); fc = json.loads(w.read_text(encoding="utf-8"))
nv = max(sum(len(r) for r in f["geometry"]["coordinates"]) for f in fc["features"])
res = {"before_cap": len(allk), "after_cap": len(capped), "written": len(fc["features"]), "max_vertices": nv,
       "mb": round(w.stat().st_size / 2**20, 2), "validate_ok": v["ok"], "errors": v["errors"][:3],
       "by_class_before": dict(Counter(f["class_name"] for f in allk)), "by_class_after": dict(Counter(f["class_name"] for f in capped)),
       "finish_all_s": round(t1 - t0, 1), "cap_write_s": round(t2 - t1, 2), "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
print(json.dumps(res, ensure_ascii=False))
Path(__file__).with_name("cap_all.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
