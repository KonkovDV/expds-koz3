import json
from collections import Counter

fc = json.load(open(r"F:\expds3\work\smoke_coarse\win\result.geojson", encoding="utf-8"))
c = Counter()
for f in fc["features"]:
    p = f["properties"]
    c[(p.get("class_name"), p.get("markup_type"), p.get("threshold_group"), round(float(p.get("confidence", 0)), 3))] += 1
print("objects", len(fc["features"]))
for k, n in c.most_common(30):
    print(n, k)
