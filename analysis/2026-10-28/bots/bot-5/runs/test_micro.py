"""Сквозная проверка post.micro_circles + post.finalize + официальная метрика на синтетических пиках (центр эталона + сдвиг)."""
import json, sys, io, contextlib, math, random
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo"); sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import compute_metrics as cm
from shapely.geometry import shape, mapping, Point
from expds import post
gt = json.load(open("gt_arch_3857.geojson"))
rng = random.Random(1)
feats = []
for f in gt["features"]:
    g = cm._extract_geometry(f)
    if g is None: continue
    c = g.centroid; s = post._ground_scale(c.y)
    for _ in range(2):  # пик у эталона со сдвигом до 1,5 м + один ложный в 10 м
        t = rng.uniform(0, 2*math.pi); d = rng.uniform(0, 1.5)
        p = Point(c.x + d*math.cos(t)/s, c.y + d*math.sin(t)/s)
        feats.append({"geom": p.buffer(2.0/s, quad_segs=16), "class_name": "arkhitektury", "confidence": rng.random(),
                      "markup_type": f["properties"]["markup_type"], "region_name": f["properties"]["region_name"]})
    p = Point(c.x + 10/s, c.y)
    feats.append({"geom": p.buffer(2.0/s, quad_segs=16), "class_name": "arkhitektury", "confidence": rng.random()*0.5,
                  "markup_type": f["properties"]["markup_type"], "region_name": f["properties"]["region_name"]})
def score(fs):
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": mapping(x["geom"]),
          "properties": {"class_name": x["class_name"], "confidence": x["confidence"]}} for x in fs]}
    with contextlib.redirect_stdout(io.StringIO()):
        r = cm.F1Calculator().calculate_f1(fc, gt)
    a = r["per_class"]["arkhitektury"]; return {k: a[k] for k in ("tp", "fp", "fn", "f1")}, r["final_score"] if "final_score" in r else None
base = post.finalize(feats)
for rad in (0.25, 0.05):
    mic = post.finalize(post.micro_circles(feats, {"classes": ["arkhitektury"], "radius_m": rad}))
    print("r=", rad, "объектов", len(base), "→", len(mic), "| 2 м:", score(base)[0], "| микро:", score(mic)[0])
from expds.watchdog import finish
cfg = {"default_threshold": 0.0, "thresholds": {}, "thresholds_by_markup": {}, "post": {"micro_circles": {"classes": ["arkhitektury"], "radius_m": 0.25}}}
out = finish([dict(f) for f in feats], cfg)
print("finish c [post.micro_circles]:", len(out), "объектов, макс. площадь арх. м² земли:",
      round(max(f["geom"].area * post._ground_scale(f["geom"].centroid.y)**2 for f in out if f["class_name"]=="arkhitektury"), 3), score(out)[0])
