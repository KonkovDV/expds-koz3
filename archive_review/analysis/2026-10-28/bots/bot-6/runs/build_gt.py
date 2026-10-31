import json, re, sys
from pathlib import Path
root = Path("/workspace/data/train-expds-3")
CLS = {"курганы_поврежденные":"kurgany_povrezhdennye","курганы_целые":"kurgany_tselye","городища":"gorodishcha",
       "фортификации":"fortifikatsii","архитектуры":"arkhitektury"}
MT = {"Li":"li","Ae":"ae","SpOr":"sp_or","Or":"or"}
feats=[]; bad=0
for f in sorted(root.glob("*/06_*разметка/*.geojson")):
    region=f.parts[-3]
    m=re.search(r"_(Li|Ae|SpOr|Or)_(.+)\.geojson$", f.name)
    if not m: bad+=1; continue
    cls=CLS.get(m.group(2))
    if not cls: bad+=1; continue
    fc=json.loads(f.read_text(encoding="utf-8"))
    crs=(fc.get("crs") or {}).get("properties",{}).get("name","")
    if "3857" not in crs and crs:
        src = "OGC:CRS84" if "CRS84" in crs else crs
        from shapely.geometry import shape, mapping
        from shapely.ops import transform as T
        from pyproj import Transformer
        tr=Transformer.from_crs(src,"EPSG:3857",always_xy=True)
        for ft in fc["features"]:
            if ft.get("geometry"): ft["geometry"]=mapping(T(tr.transform, shape(ft["geometry"])))
    for ft in fc["features"]:
        if ft.get("geometry"):
            feats.append({"type":"Feature","geometry":ft["geometry"],"properties":{"region_name":region,"markup_type":MT[m.group(1)],"class_name":cls}})
Path(sys.argv[1]).write_text(json.dumps({"type":"FeatureCollection","features":feats},ensure_ascii=False),encoding="utf-8")
print(len(feats),"features; skipped files",bad)
