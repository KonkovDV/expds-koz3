"""Эталон архитектуры (и всех классов) из /workspace/data/train-expds-3 → единый FeatureCollection EPSG:3857."""
import json, re, sys, glob, os
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform
sys.path.insert(0, "/workspace/koz3_shared/bot-5/repo/materials")
import compute_metrics as cm

ROOT = Path("/workspace/data/train-expds-3")
MT = {"li": "li", "ae": "ae", "ае": "ae", "spor": "sp_or", "or": "or", "maggr": "maggr", "etgr": "etgr", "terfl": "terfl"}

def crs_of(d):
    n = (d.get("crs") or {}).get("properties", {}).get("name", "EPSG:4326")
    m = re.search(r"EPSG::?(\d+)", n)
    return int(m.group(1)) if m else 4326

def load(classes=None):
    feats = []
    for f in sorted(ROOT.glob("*/06_*разметка/*.geojson")):
        region = f.parts[-3]
        stem = f.stem
        m = re.search(r"_(Li|Ae|Ае|SpOr|Or|MagGr|EtGr|TerFl)_(.+)$", stem, re.I)
        if not m:
            continue
        mt = MT[m.group(1).lower()]
        cls = cm.normalize_class_name(m.group(2).lower())
        if cls is None or cls not in cm.CLASS_WEIGHTS:
            continue
        if classes and cls not in classes:
            continue
        d = json.load(open(f, encoding="utf-8"))
        epsg = crs_of(d)
        tr = None if epsg == 3857 else Transformer.from_crs(epsg, 3857, always_xy=True).transform
        for i, ft in enumerate(d.get("features", [])):
            g = ft.get("geometry")
            if not g:
                continue
            try:
                geom = shape(g)
            except Exception:
                continue
            if tr:
                geom = transform(tr, geom)
            feats.append({"type": "Feature", "geometry": mapping(geom),
                          "properties": {"region_name": region, "class_name": cls, "markup_type": mt,
                                         "file": str(f.relative_to(ROOT)), "idx": i}})
    return {"type": "FeatureCollection", "features": feats}

if __name__ == "__main__":
    fc = load()
    json.dump(fc, open("gt_all_3857.geojson", "w"), ensure_ascii=False)
    arch = {"type": "FeatureCollection", "features": [f for f in fc["features"] if f["properties"]["class_name"] == "arkhitektury"]}
    json.dump(arch, open("gt_arch_3857.geojson", "w"), ensure_ascii=False)
    from collections import Counter
    print(Counter((f["properties"]["class_name"], f["properties"]["markup_type"], f["geometry"]["type"]) for f in fc["features"]))
