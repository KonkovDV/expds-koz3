"""Синтетика: один курган размечен в слоях li и ae (два GT-объекта); выдача 1 или 2 круга, с markup_type и без."""
import json, sys, subprocess, itertools
from pathlib import Path
x0, y0 = 4180000.0, 7500000.0   # EPSG:3857
def circle(cx, cy, r=8, n=32):
    import math
    return {"type":"Polygon","coordinates":[[[cx+r*math.cos(2*math.pi*k/n), cy+r*math.sin(2*math.pi*k/n)] for k in range(n)]+[[cx+r, cy]]]}
def feat(mt, conf=None, dx=0.0, cls="kurgany_tselye"):
    p={"region_name":"r1","class_name":cls}
    if mt: p["markup_type"]=mt
    if conf is not None: p["confidence"]=conf
    return {"type":"Feature","properties":p,"geometry":circle(x0+dx,y0)}
def fc(fs): return {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:3857"}},"features":fs}
# второй курган, чтобы F1 не вырождался: только li
gt=fc([feat("li"), feat("ae"), feat("li", dx=200)])
cases={
 "A_1круг_без_типа":[feat(None,0.9), feat(None,0.9,dx=200)],
 "B_2круга_без_типа":[feat(None,0.9), feat(None,0.8), feat(None,0.9,dx=200)],
 "C_1круг_li":[feat("li",0.9), feat("li",0.9,dx=200)],
 "D_2круга_li+ae":[feat("li",0.9), feat("ae",0.8), feat("li",0.9,dx=200)],
 "E_2круга_оба_li":[feat("li",0.9), feat("li",0.8), feat("li",0.9,dx=200)],
 "F_3круга_li+ae+лишний":[feat("li",0.9), feat("ae",0.8), feat("ae",0.7), feat("li",0.9,dx=200)],
}
d=Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
(d/"gt.geojson").write_text(json.dumps(gt))
res={}
for name, fs in cases.items():
    p=d/f"{name}.geojson"; p.write_text(json.dumps(fc(fs)))
    o=d/f"{name}.json"
    subprocess.run([sys.executable,"repo/materials/compute_metrics.py","-p",str(p),"-g",str(d/"gt.geojson"),"--output-json",str(o)],check=True,capture_output=True)
    r=json.loads(o.read_text())
    pc=r.get("per_class",r.get("class_results",{})).get("kurgany_tselye",{})
    res[name]={"final":round(r["final_score"],6),**{k:pc.get(k) for k in ("tp","fp","fn")},
               "by_data_type":r.get("auxiliary",{}).get("by_data_type")}
    print(name, res[name]["final"], pc.get("tp"), pc.get("fp"), pc.get("fn"))
(d/"synth_result.json").write_text(json.dumps(res,ensure_ascii=False,indent=1))
