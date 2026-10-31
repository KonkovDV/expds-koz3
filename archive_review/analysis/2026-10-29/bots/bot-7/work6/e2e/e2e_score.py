"""bot-7, 29.10: сквозное сравнение двух выдач main.py (v2 и v3) против эталона официальной метрикой.
python e2e_score.py --metric compute_metrics.py --gt gt.geojson --a v2=res_v2.geojson --b v3=res_v3.geojson --out e2e.json
Считает: всё вместе, по регионам и по markup_type (только регионы, которые есть в эталоне). Ничего не подбирает."""
import argparse, collections, importlib.util, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--metric", required=True); ap.add_argument("--gt", required=True)
ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.metric).parent))
spec = importlib.util.spec_from_file_location("cm", a.metric); cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
runs = {}
for s in (a.a, a.b):
    name, p = s.split("=", 1); runs[name] = json.loads(Path(p).read_text(encoding="utf-8"))
regions = sorted({f["properties"]["region_name"] for f in gt["features"]})

def fc(feats): return {"type": "FeatureCollection", "features": feats}
def sel(d, reg=None, mt=None):
    return [f for f in d["features"] if (reg is None or f["properties"].get("region_name") == reg)
            and (mt is None or f["properties"].get("markup_type") == mt)]
def score(p, g):
    if not g: return None
    return cm.F1Calculator().calculate_f1(fc(p), fc(g))["final_score"]

res = {"regions": regions, "rows": []}
gfe = [f for f in gt["features"]]
mts = sorted({f["properties"]["markup_type"] for f in gfe})
for reg in [None] + regions:
    for mt in [None] + mts:
        g = sel(gt, reg, mt)
        if not g: continue
        row = {"region": reg or "ALL", "markup": mt or "ALL", "gt": len(g)}
        for n, d in runs.items():
            p = [f for f in sel(d, reg, mt) if reg is not None or f["properties"].get("region_name") in regions]
            row[f"n_{n}"] = len(p); row[f"f1_{n}"] = score(p, g)
        names = list(runs)
        if row[f"f1_{names[0]}"] is not None: row["delta"] = row[f"f1_{names[1]}"] - row[f"f1_{names[0]}"]
        res["rows"].append(row); print(json.dumps(row, ensure_ascii=False))
for n, d in runs.items():
    res[f"counts_{n}"] = collections.Counter(
        f"{f['properties'].get('markup_type')}/{f['properties'].get('class_name')}" for f in d["features"])
Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
