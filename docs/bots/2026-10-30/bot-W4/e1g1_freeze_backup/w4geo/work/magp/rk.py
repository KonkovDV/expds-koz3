import sys, json, collections, numpy as np
sys.path.insert(0, "/workspace/koz3/wt/w4-pix"); sys.path.insert(0, "/workspace/koz3/wt/w4-pix/scripts")
from expds import metric; import pix_clean_eval as E
W="/workspace/koz3/w4geo/work/"
gt_p="/workspace/koz3/w5data/gt_train_v2.geojson"
gt = E._load(gt_p, None, "maggr"); regs = {f["properties"]["region_name"] for f in gt["features"]}; gt = E._load(gt_p, regs, "maggr")
def cv(p): return metric.build_curves(E._load(p, regs, "maggr"), gt)["finds_points"]
c=cv(W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1_M3.geojson"); h=cv(W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1_M3_ht.geojson"); rf=cv(W+"../ref/out/REF_v2.geojson")
print(type(c), [a for a in dir(c) if not a.startswith('_')])
nr=collections.Counter(rf.pred_region); nh=collections.Counter(h.pred_region)
pr=np.array(c.pred_region)
for r in sorted(c.gt_regions):
    m=pr==r; cf=c.conf[m]; ht=c.hit[m]; o=np.argsort(-cf); cum=np.cumsum(ht[o]); g=c.gt_regions[r]
    def at(n): n=min(n,len(o)); return (cum[n-1]/g if n else 0, cum[n-1]/n if n else 0, cf[o][n-1] if n else 1)
    tr=sum(rf.hit[np.array(rf.pred_region)==r]); 
    print(f"{r[:24]:24s} gt={g:4d} ours_ht n={nh[r]:4d} R@nht={at(nh[r])[0]:.3f} | REF n={nr[r]:4d} R={tr/g:.3f} P={tr/max(1,nr[r]):.3f} | ours@REFn R={at(nr[r])[0]:.3f} P={at(nr[r])[1]:.3f} conf={at(nr[r])[2]:.3f} | @2xREF R={at(2*nr[r])[0]:.3f} P={at(2*nr[r])[1]:.3f}")
