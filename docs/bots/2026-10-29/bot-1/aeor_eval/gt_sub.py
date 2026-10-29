import json, sys
sys.path.insert(0, "/workspace/koz3_shared/bot-10/repo")
from expds.naming import site_family
T = json.load(open("touched.json")); mem = set(T["member_families"])
gt = json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))["features"]
sub = [f for f in gt if site_family(f["properties"]["region_name"]) in mem]
json.dump({"type": "FeatureCollection", "features": sub}, open("gt_t.geojson", "w", encoding="utf-8"), ensure_ascii=False)
import collections
print(len(sub), collections.Counter(f["properties"].get("markup_type") for f in sub))
