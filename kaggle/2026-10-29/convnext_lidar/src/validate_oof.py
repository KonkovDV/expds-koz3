# Проверка OOF convnext_lidar: схема = /workspace/rerank/stack_arch.geojson (fid, class_name, confidence, region_name, crs,
# markup_type, original_crs), Polygon EPSG:3857, классы КОЗ-3; метрика — официальный compute_metrics.py (koz3-1.5.0);
# контроль формата: «идеальные» круги r=1 м в точке GT (representative_point) должны дать ~1.0.
# python validate_oof.py OOF.geojson GT.geojson [compute_metrics.py]
import json, sys, subprocess, tempfile, collections, pathlib
from shapely.geometry import shape, mapping, Point
from shapely.ops import transform
from pyproj import Transformer
oof, gt = sys.argv[1], sys.argv[2]
import os; R = float(os.environ.get("VAL_R", "1.0"))           # радиус «идеальных» кругов контроля
cm = sys.argv[3] if len(sys.argv) > 3 else str(pathlib.Path(__file__).parent / "vendor/compute_metrics.py")
KEYS = {"fid", "class_name", "confidence", "region_name", "crs", "markup_type", "threshold_group", "original_crs"}   # threshold_group=li_cnx (29.10)
CLS = {"kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"}
P = json.load(open(oof, encoding="utf-8")); bad = collections.Counter(); cls = collections.Counter(); fids = set()
for f in P["features"]:
    p = f["properties"]
    if set(p) != KEYS: bad["keys"] += 1
    if f["geometry"]["type"] != "Polygon": bad["geom"] += 1
    if p["crs"] != "EPSG:3857" or p["markup_type"] != "li" or p.get("threshold_group") != "li_cnx" or not str(p["original_crs"]).startswith("EPSG:32"): bad["crs"] += 1
    if p["class_name"] not in CLS: bad["class"] += 1
    if not (0 <= p["confidence"] <= 1): bad["conf"] += 1
    x, y = f["geometry"]["coordinates"][0][0]
    if not (-2.1e7 < x < 2.1e7 and 0 < y < 2e7): bad["coord"] += 1
    fids.add(p["fid"]); cls[p["class_name"]] += 1
if len(fids) != len(P["features"]): bad["fid_dup"] += 1
print("OOF:", len(P["features"]), "объектов", dict(cls), "регионы", sorted({f["properties"]["region_name"] for f in P["features"]}))
print("ошибки схемы:", dict(bad) or "нет")
def run(pred):
    r = subprocess.run([sys.executable, cm, "-p", pred, "-g", gt], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "ERR " + r.stderr[-400:]
print("score OOF (compute_metrics):", run(oof))
G = json.load(open(gt, encoding="utf-8")); perf = []
for i, f in enumerate(G["features"]):
    g = shape(f["geometry"]); q = P["features"][0]["properties"] if P["features"] else {}
    oc = next((x["properties"]["original_crs"] for x in P["features"] if x["properties"]["region_name"] == f["properties"]["region_name"]), "EPSG:32636")
    to = Transformer.from_crs("EPSG:3857", oc, always_xy=True).transform; back = Transformer.from_crs(oc, "EPSG:3857", always_xy=True).transform
    pt = transform(to, g).representative_point()
    perf.append(dict(type="Feature", geometry=mapping(transform(back, pt.buffer(R, quad_segs=4))), properties=dict(
        fid=i, class_name=f["properties"]["class_name"], confidence=1.0, region_name=f["properties"]["region_name"],
        crs="EPSG:3857", markup_type="li", threshold_group="li_cnx", original_crs=oc)))
with tempfile.NamedTemporaryFile("w", suffix=".geojson", delete=False) as t:
    json.dump(dict(type="FeatureCollection", crs=P.get("crs"), features=perf), t)
print(f"score идеальных кругов r={R} м (контроль формата):", run(t.name), "на", len(perf), "объектах GT")
