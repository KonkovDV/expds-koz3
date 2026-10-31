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
        rows.append((pr.get("region_name"), pr.get("markup_type"), pr.get("class_name"),
                     round(float(pr.get("confidence", 0)), 6), round(c[0], 2), round(c[1], 2),
                     pr.get("threshold_group")))
    return rows


a = load(r"F:\expds3\work\smoke_win\result.geojson")
b = load(r"F:\expds3\work\smoke_dock\result.geojson")
print("win", len(a), "dock", len(b))
print("win", Counter((r, m, c) for r, m, c, *_ in a))
print("dock", Counter((r, m, c) for r, m, c, *_ in b))
ka = {(r, m, c, x, y): conf for r, m, c, conf, x, y, g in a}
kb = {(r, m, c, x, y): conf for r, m, c, conf, x, y, g in b}
print("only win", len(set(ka) - set(kb)), "only dock", len(set(kb) - set(ka)))
diffs = [abs(ka[k] - kb[k]) for k in set(ka) & set(kb) if ka[k] != kb[k]]
print("matched", len(set(ka) & set(kb)), "conf diffs", len(diffs), "max", max(diffs) if diffs else 0)
print("groups win", Counter(g for *_, g in a))
print("groups dock", Counter(g for *_, g in b))
