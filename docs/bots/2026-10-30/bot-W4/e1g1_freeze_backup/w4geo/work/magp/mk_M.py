#!/usr/bin/env python3
"""exec A 20:45 MSK: MagGr finds P/R lift on E1 (GT-free, test-time rules). Only maggr finds_points confidence changes.
usage: mk_M.py NAME Q A D X   (A=1 -> no per-raster norm)
  1) per-raster conf norm: c' = 0.9 * c^A * (c/q_Q(raster))^(1-A), clip [0,1]   (q_Q over that raster's MagGr finds candidates)
  2) soft-NMS: greedy by c' desc, each lower-conf MagGr find within D m (same region, any MagGr raster) gets c' *= X
-> cstar/C_CetO3_Sr8G_noTerFl_L_E1_NAME.geojson"""
import sys,json
sys.path.insert(0,"/workspace/koz3/w4geo/work/magp")
from feats import *
N=sys.argv[1]; Q,A,D,X=map(float,sys.argv[2:6])
full=json.load(open(BASE)); idx=[i for i,f in enumerate(full["features"]) if f["properties"].get("markup_type")=="maggr" and f["properties"]["class_name"]=="finds_points"]
F=[full["features"][i] for i in idx]; df=table(F); c=df.conf.copy()
if A<1:
    for s in np.unique(df.src):
        m=df.src==s; v=c[m]; c[m]=np.clip(0.9*v**A*(v/np.quantile(v,Q))**(1-A),0,1)
c,_=nms(DF(**{**df,"conf":c}),D,1,1,soft=X)
for i,cc in zip(idx,c): full["features"][i]["properties"]["confidence"]=round(float(cc),6)
out=W+f"cstar/C_CetO3_Sr8G_noTerFl_L_E1_{N}.geojson"; json.dump(full,open(out,"w"),ensure_ascii=False); print(out,len(full["features"]),len(idx))
