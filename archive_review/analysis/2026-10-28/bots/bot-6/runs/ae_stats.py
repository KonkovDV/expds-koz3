import json, collections, numpy as np
from shapely.geometry import shape
from shapely import STRtree
from pyproj import Transformer
fc=json.load(open("work/gt_train_local.geojson",encoding="utf-8"))
by=collections.defaultdict(lambda: collections.defaultdict(list))
for f in fc["features"]:
    p=f["properties"]; by[p["region_name"]][(p["markup_type"],p["class_name"])].append(shape(f["geometry"]))
K=("kurgany_povrezhdennye","kurgany_tselye")
tot=collections.Counter(); radii=collections.defaultdict(list)
rows=[]
for r,L in sorted(by.items()):
    ae={c:L.get(("ae",c),[]) for c in ("kurgany_povrezhdennye","kurgany_tselye","gorodishcha","fortifikatsii","arkhitektury")}
    if not any(ae.values()): continue
    li=[g for c in K for g in L.get(("li",c),[])]
    aek=[g for c in K for g in ae[c]]
    # metric scale: 3857 -> local scale factor
    lat=None
    if aek:
        x,y=aek[0].representative_point().coords[0]
        lon,lat=Transformer.from_crs(3857,4326,always_xy=True).transform(x,y)
    k=np.cos(np.radians(lat)) if lat is not None else 1
    for c in K:
        for g in ae[c]: radii[("ae",c)].append(np.sqrt(g.area*k*k/np.pi))
        for g in L.get(("li",c),[]): radii[("li",c)].append(np.sqrt(g.area*k*k/np.pi))
    # ae kurgans that have any li kurgan whose rep point lies inside (what a li->ae copy could hit, ignoring class)
    reach=0
    if li and aek:
        t=STRtree([g.representative_point() for g in li])
        for g in aek:
            if len(t.query(g, predicate="contains")): reach+=1
    row=dict(region=r, li_kurgans=len(li), **{c[:12]:len(v) for c,v in ae.items()}, ae_kurg_with_li_point_inside=reach)
    rows.append(row)
    for c,v in ae.items(): tot[c]+=len(v)
    tot["ae_kurg"]+=len(aek); tot["reach"]+=reach; tot["ae_kurg_in_regions_without_li"]+= 0 if li else len(aek)
for row in rows: print(row)
print(dict(tot))
for key,v in radii.items(): print(key, "n",len(v),"median eq.radius m", round(float(np.median(v)),2), "p10/p90", np.round(np.percentile(v,[10,90]),2))
json.dump({"rows":rows,"totals":tot,"radii":{f"{a}:{b}":[len(v),float(np.median(v))] for (a,b),v in radii.items()}},open("work/ae_gt_stats.json","w"),ensure_ascii=False,indent=1)
