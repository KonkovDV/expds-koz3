"""sample sigma-1 smoothed OOF prob of each probs dir at E1 MagGr finds points -> night/s1_<probs>.npy"""
import sys, pickle, os
sys.path.insert(0,"/workspace/koz3/w4geo/work/magp")
from feats import *
from scipy import ndimage as ndi
from pyproj import Transformer
d,F,O=split(W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson"); df=table(F)
cx=np.array([shape(f["geometry"]).centroid.coords[0] for f in F])
for PROBS in sys.argv[1:]:
    out=np.full(len(F),np.nan)
    for s in np.unique(df.src):
        m=np.flatnonzero(df.src==s); reg=df.reg[m[0]]; fz=W+f"{PROBS}/{reg}/{os.path.splitext(s)[0]}.npz"
        if not os.path.exists(fz): continue
        z=np.load(fz); prob=z["prob"].astype(np.float32); pp=pickle.load(open(fz.replace(".npz",".prep.pkl"),"rb"))
        X,Y=Transformer.from_crs("EPSG:3857",pp["crs"],always_xy=True).transform(cx[m,0],cx[m,1]); cc,rr=(~pp["transform"])*(X,Y)
        r=np.clip(np.round(rr).astype(int),0,prob.shape[0]-1); c=np.clip(np.round(cc).astype(int),0,prob.shape[1]-1)
        sm=ndi.gaussian_filter(prob,1.0,mode="nearest")
        lm=ndi.maximum_filter(sm,size=3)  # tolerate 1px misalign
        out[m]=lm[r,c]
    np.save(W+f"night/s1_{PROBS}.npy",out); print(PROBS, np.isnan(out).sum(), flush=True)
