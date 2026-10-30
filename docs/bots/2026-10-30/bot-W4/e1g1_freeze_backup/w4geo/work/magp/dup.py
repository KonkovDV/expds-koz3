from lib import *
from shapely.geometry import shape
from scipy.spatial import cKDTree
regs=regions(); gt=E._load(GTP,regs,"maggr")
for nm in ["C_CetO3_Sr8G_noTerFl_L_E1_ht","C_CetO3_Sr8G_noTerFl_L_E1"]:
    P=E._load(W+f"cstar/{nm}.geojson", regs, "maggr"); h,_=feat_hits(P,gt)
    fs=[(f,hh) for f,hh in zip(P["features"],h) if hh is not None]
    xy=np.array([mod._to_utm(shape(f["geometry"]),"auto").centroid.coords[0] for f,_ in fs])
    cf=np.array([f["properties"]["confidence"] for f,_ in fs]); hv=np.array([hh for _,hh in fs])
    rg=np.array([f["properties"]["region_name"] for f,_ in fs]); sr=np.array([f["properties"]["source"] for f,_ in fs])
    t=cKDTree(xy)
    for d in (1.0,1.5,2.0,3.0):
        dx=0; dfp=0; dsame=0
        for i,j in t.query_pairs(d):
            if rg[i]!=rg[j]: continue
            lo=i if cf[i]<cf[j] else j
            if sr[i]!=sr[j]: dx+=1; dfp+= (not hv[lo])
            else: dsame+=1; dfp+=0; globals().setdefault("sfp",{}); sfp[(nm,d)]=sfp.get((nm,d),0)+(not hv[lo])
        print(nm[-12:], d, "xraster pairs",dx,"lower-is-FP",dfp,"same-raster pairs",dsame, "n",len(fs),"tp",hv.sum())
print(sfp)
