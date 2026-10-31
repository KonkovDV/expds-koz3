"""MagGr finds P-R curve (global conf order, metric P=tp/(tp+.5fp)) — REF vs ours raw (operating-point check)"""
import sys; sys.path.insert(0,"/workspace/koz3/w4geo/work/magp")
from ev import *
for p in sys.argv[1:]:
    d=E._load(p,regs); F=[f for f in d["features"] if f["properties"]["class_name"]=="finds_points" and f["properties"].get("markup_type","maggr")=="maggr"]
    c=metric.build_curves({"type":"FeatureCollection","features":F},GTF)["finds_points"]
    o=np.argsort(-c.conf,kind="stable"); h=c.hit[o].astype(float); tp=np.cumsum(h); fp=np.cumsum(1-h)
    P=tp/(tp+.5*fp); R=tp/c.n_gt; F1=2*P*R/(P+R+1e-12); k=F1.argmax()
    s=f"{p.split('/')[-1][:40]:40s} n{len(F)} Rmax {R[-1]:.3f} F1opt {F1[k]:.3f}@P{P[k]:.3f}/R{R[k]:.3f}"
    for t in (.70,.73,.76,.80,.83,.859,.88):
        i=np.searchsorted(R,t); s+=f" | R{t}:P{P[i]:.3f}" if i<len(R) else f" | R{t}:-"
    print(s,flush=True)
