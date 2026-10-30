#!/usr/bin/env python3
"""mk_M.py generalised: IN OUT Q A D X [ENS_RATIO_NPY]  (per-raster norm + soft-NMS on maggr finds conf; GT-free)"""
import sys,json
sys.path.insert(0,"/workspace/koz3/w4geo/work/magp")
from feats import *
IN,OUT=sys.argv[1:3]; Q,A,D,X=map(float,sys.argv[3:7])
full=json.load(open(IN)); idx=[i for i,f in enumerate(full["features"]) if f["properties"].get("markup_type")=="maggr" and f["properties"]["class_name"]=="finds_points"]
F=[full["features"][i] for i in idx]; df=table(F); c=df.conf.astype(float).copy()
if len(sys.argv)>7: r=np.load(sys.argv[7]); assert len(r)==len(c); c=np.clip(c*r,0,1)
if A<1:
    for s in np.unique(df.src):
        m=df.src==s; v=c[m]; c[m]=np.clip(0.9*v**A*(v/np.quantile(v,Q))**(1-A),0,1)
if X<1: c,_=nms(DF(**{**df,"conf":c}),D,1,1,soft=X)
for i,cc in zip(idx,c): full["features"][i]["properties"]["confidence"]=round(float(cc),6)
json.dump(full,open(OUT,"w"),ensure_ascii=False); print(OUT,len(full["features"]),len(idx))
