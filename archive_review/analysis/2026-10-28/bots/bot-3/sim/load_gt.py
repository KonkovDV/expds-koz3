"""Загрузка эталона геофизики (MagGr/EtGr/TerFl) из /workspace/data/train-expds-3 в UTM участка."""
import json, re, glob, os
from pathlib import Path
from shapely.geometry import shape
from shapely.validation import make_valid
from pyproj import CRS, Transformer
from shapely.ops import transform as stf

ROOT = Path("/workspace/data/train-expds-3")

def utm_of(region_dir):
    d = json.loads((region_dir / "UTM.json").read_text(encoding="utf-8-sig"))
    return CRS.from_user_input(d["crs"])

def family(region):
    return re.split(r"[_\s]", region.split("_", 1)[1], maxsplit=1)[0]

def modality(name):
    n = name.lower()
    for m in ("maggr", "etgr", "terfl"):
        if m in n: return m
    return None

def load():
    units = []  # one unit = one annotation file (layer)
    for f in sorted(ROOT.glob("*/06_*разметка/*.geojson")):
        mt = modality(f.name)
        if mt is None: continue
        cls = "finds_points" if "findspoints" in f.name.lower() else ("object_poly" if "objectpoly" in f.name.lower() else None)
        if cls is None: continue
        reg = f.parts[len(ROOT.parts)]
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        utm = utm_of(ROOT / reg)
        src = d.get("crs", {}).get("properties", {}).get("name")
        src = CRS.from_user_input(src) if src else utm
        tr = None if src == utm else Transformer.from_crs(src, utm, always_xy=True).transform
        geoms = []
        for ft in d.get("features", []):
            if not ft.get("geometry"): continue
            g = shape(ft["geometry"])
            if tr: g = stf(tr, g)
            if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_valid: g = make_valid(g)
            parts = list(getattr(g, "geoms", [g]))
            for p in parts:
                if p.geom_type in ("Polygon", "Point") and not p.is_empty: geoms.append(p)
        units.append(dict(region=reg, family=family(reg), markup=mt, cls=cls, file=f.name, geoms=geoms, crs=str(utm)))
    return units

if __name__ == "__main__":
    from collections import Counter
    U = load()
    c = Counter()
    for u in U:
        c[(u["family"], u["markup"], u["cls"])] += len(u["geoms"])
    for k, v in sorted(c.items()): print(k, v)
    print("files", len(U), "types", Counter(g.geom_type for u in U for g in u["geoms"]))
