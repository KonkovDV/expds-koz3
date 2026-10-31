import json
from collections import Counter


def load(path):
    fc = json.load(open(path, encoding="utf-8"))
    rows = []
    for f in fc["features"]:
        pr = f["properties"]
        c = f["geometry"]["coordinates"]
        while c and isinstance(c[0], list):
            c = c[0]
        rows.append((pr.get("class_name"), pr.get("markup_type"),
                     round(float(pr.get("confidence", 0)), 5), round(c[0], 1), round(c[1], 1)))
    return rows


a = load(r"F:\expds3\work\smoke_coarse\win\result.geojson")
b = load(r"F:\expds3\work\smoke_coarse\dock\result.geojson")
print("win", len(a), "dock", len(b), Counter(x[0] for x in a), Counter(x[0] for x in b))
ka = {r[1:]: r[0] for r in [(conf, cls, x, y) for cls, mt, conf, x, y in a]}
# key by class, xy
ka = {(cls, x, y): conf for cls, mt, conf, x, y in a}
kb = {(cls, x, y): conf for cls, mt, conf, x, y in b}
print("only win", len(set(ka) - set(kb)), "only dock", len(set(kb) - set(ka)))
diffs = [abs(ka[k] - kb[k]) for k in set(ka) & set(kb) if ka[k] != kb[k]]
print("matched", len(set(ka) & set(kb)), "max conf diff", max(diffs) if diffs else 0)
