import json,sys,numpy as np
from shapely.geometry import shape
for p in sys.argv[1:]:
    F=json.load(open(p))["features"]
    m=[f for f in F if f["properties"].get("markup_type")=="maggr" and f["properties"].get("class_name")=="finds_points"]
    if not m: m=[f for f in F if f["properties"].get("class_name")=="finds_points"]
    a=np.array([shape(f["geometry"]).area for f in m]); t=collections=None
    gt=collections
    types={}
    for f in m: types[f["geometry"]["type"]]=types.get(f["geometry"]["type"],0)+1
    c=np.array([float(f["properties"].get("confidence",np.nan)) for f in m])
    print(p.split('/')[-1],len(m),types,"area q10/50/90",np.round(np.quantile(a,[.1,.5,.9]),3),"conf q",np.round(np.nanquantile(c,[.1,.5,.9]),3), "props",list(m[0]["properties"].keys()))
