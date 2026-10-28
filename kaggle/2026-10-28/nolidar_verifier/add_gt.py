"""Компактная таблица для встраивания: кандидаты (kind=cand) + GT sp_or без лидара (kind=gt, только для обучения с джиттером)."""
import csv, gzip, json, base64
from shapely.geometry import shape
LAS = set(json.load(open("/workspace/expds-koz3/notes/las_boxes_59.json")))
GROUP = {"kurgany_povrezhdennye": "kurgan", "kurgany_tselye": "kurgan", "gorodishcha": "gor"}
CC = {"kurgany_povrezhdennye": "P", "kurgany_tselye": "T", "gorodishcha": "G"}
out = [["kind", "stack_idx", "region", "markup_type", "cls", "confidence", "cx", "cy", "size_3857", "label"]]
for r in csv.DictReader(gzip.open("candidates_nolidar.csv.gz", "rt", encoding="utf-8")):
    out.append(["c", r["stack_idx"], r["region"], r["markup_type"], CC[r["class_name"]], round(float(r["confidence"]), 4), round(float(r["cx"]), 1), round(float(r["cy"]), 1), round(float(r["size_3857"]), 1), r["label"]])
ng = 0
for f in json.load(open("/workspace/rerank/gt_train_v2.geojson", encoding="utf-8"))["features"]:
    p = f["properties"]; g = GROUP.get(p["class_name"])
    if g and p["region_name"] not in LAS and p["markup_type"] == "sp_or" and f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
        s = shape(f["geometry"]).buffer(0); c = s.representative_point(); b = s.bounds
        out.append(["g", -1, p["region_name"], "sp_or", CC[p["class_name"]], 1, round(c.x, 1), round(c.y, 1), round(max(b[2]-b[0], b[3]-b[1]), 1), 1]); ng += 1
import io
buf = io.StringIO(); csv.writer(buf).writerows(out)
raw = gzip.compress(buf.getvalue().encode("utf-8"), 9)
open("table_embed.b64", "w").write(base64.b64encode(raw).decode())
print("rows", len(out) - 1, "gt", ng, "gz", len(raw), "b64", len(base64.b64encode(raw)))
