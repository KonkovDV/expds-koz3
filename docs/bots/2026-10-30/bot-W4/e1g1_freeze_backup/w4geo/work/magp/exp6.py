from feats import *
d,F,O=split(); df=table(F); c0=df.conf
A=E._load(W+"A18.geojson",regs,"maggr"); AF=[f for f in A["features"] if f["properties"]["class_name"]=="finds_points"]
print("A18 maggr finds",len(AF), collections.Counter(metric.threshold_group(f["properties"]) for f in AF).most_common(5))
ha,_=feat_hits({"type":"FeatureCollection","features":AF},GTF)
print("A18 alone hits",sum(ha), collections.Counter(f["properties"]["region_name"][:3] for f,h in zip(AF,ha) if h))
adf=table(AF)
t=cKDTree(df.xy())
# classic preds not near pix (2m)
dd,_=t.query(adf.xy()); new=dd>2.0
print("classic new (>2m from any pix cand):",new.sum(),"hits among new", sum(h for h,nw in zip(ha,new) if nw), collections.Counter(r[:3] for r,nw,h in zip(adf.reg,new,ha) if nw and h))
score(F,"E1")
for gname in ("maggr_cls","same"):
    for dmin in (1.0,2.0):
        nw=dd>dmin; add=[]
        for f,k in zip(AF,nw):
            if not k: continue
            p={**f["properties"]}; p["threshold_group"]="maggr_cls" if gname=="maggr_cls" else p.get("threshold_group","maggr")
            p["source"]=p.get("source","classic"); add.append({"type":"Feature","geometry":f["geometry"],"properties":p})
        score(F+add,f"E1+classic new>{dmin}m grp={gname} (+{len(add)})")
        cs,_=nms(df,2.0,1,1,soft=0.7); score(with_conf(F,cs)+add,f"soft2x.7+classic new>{dmin}m grp={gname}")
