"""Score one result.geojson vs gt.geojson with official compute_metrics; emit scores.json rows."""
from __future__ import annotations
import argparse, collections, importlib.util, json, sys
from pathlib import Path

def load_cm(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("cm", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def fc(feats):
    return {"type": "FeatureCollection", "features": feats}

def sel(d, reg=None, mt=None):
    return [f for f in d["features"]
            if (reg is None or f["properties"].get("region_name") == reg)
            and (mt is None or f["properties"].get("markup_type") == mt)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="cand")
    a = ap.parse_args()
    cm = load_cm(Path(a.metric))
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    pred = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    regions = sorted({f["properties"]["region_name"] for f in gt["features"]})
    mts = sorted({f["properties"]["markup_type"] for f in gt["features"]})
    name = a.name

    def score(p, g):
        if not g:
            return None
        return cm.F1Calculator().calculate_f1(fc(p), fc(g))["final_score"]

    rows = []
    for reg in [None] + regions:
        for mt in [None] + mts:
            g = sel(gt, reg, mt)
            if not g:
                continue
            p = [f for f in sel(pred, reg, mt)
                 if reg is not None or f["properties"].get("region_name") in regions]
            row = {
                "region": reg or "ALL",
                "markup": mt or "ALL",
                "gt": len(g),
                f"n_{name}": len(p),
                f"f1_{name}": score(p, g),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    all_row = next(r for r in rows if r["region"] == "ALL" and r["markup"] == "ALL")
    out = {
        "regions": regions,
        "markups": mts,
        "rows": rows,
        f"ALL_F1_{name}": all_row[f"f1_{name}"],
        f"counts_{name}": dict(collections.Counter(
            f"{f['properties'].get('markup_type')}/{f['properties'].get('class_name')}"
            for f in pred["features"]
            if f["properties"].get("region_name") in regions
        )),
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
