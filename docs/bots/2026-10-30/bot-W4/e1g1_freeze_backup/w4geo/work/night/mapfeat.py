"""exec A night 30->31 MSK: GT-free prob-map features at each E1 MagGr finds point (OOF maps probs_n40r8).
-> night/mf_<probs>.pkl : DataFrame-like dict aligned with E1 maggr finds order (split())."""
import sys, pickle, glob, os
sys.path.insert(0,"/workspace/koz3/w4geo/work/magp")
from feats import *
from scipy import ndimage as ndi
from pyproj import Transformer
PROBS=sys.argv[1] if len(sys.argv)>1 else "probs_n40r8"
d,F,O=split(W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson"); df=table(F)
cx=np.array([shape(f["geometry"]).centroid.coords[0] for f in F])
out={k:np.full(len(F),np.nan) for k in ["s0","s1","s2","s4","m1","m2","ring","prom","hmA","lmax1","bg","vfrac"]}
for s in np.unique(df.src):
    m=np.flatnonzero(df.src==s); reg=df.reg[m[0]]
    stem=os.path.splitext(s)[0]; fz=W+f"{PROBS}/{reg}/{stem}.npz"
    if not os.path.exists(fz): print("missing",fz); continue
    z=np.load(fz); prob=z["prob"].astype(np.float32); valid=z["valid"]; pp=pickle.load(open(fz.replace(".npz",".prep.pkl"),"rb"))
    st=pp["step"]; tr=Transformer.from_crs("EPSG:3857",pp["crs"],always_xy=True)
    X,Y=tr.transform(cx[m,0],cx[m,1]); inv=~pp["transform"]; cc,rr=inv*(X,Y)
    r=np.clip(np.round(rr).astype(int),0,prob.shape[0]-1); c=np.clip(np.round(cc).astype(int),0,prob.shape[1]-1)
    px=lambda meters: max(1.0,meters/st)
    S={k:ndi.gaussian_filter(prob,sg,mode="nearest") if sg>0 else prob for k,sg in [("s0",0),("s1",1.0),("s2",2.0),("s4",4.0)]}
    for k in S: out[k][m]=S[k][r,c]
    def disc_mean(rad_m):
        R=int(round(px(rad_m))); yy,xx=np.mgrid[-R:R+1,-R:R+1]; k=((yy*yy+xx*xx)<=R*R).astype(np.float32); k/=k.sum()
        return ndi.convolve(prob,k,mode="nearest")
    m1=disc_mean(1.0); m3=disc_mean(3.0); m2=disc_mean(2.0)
    out["m1"][m]=m1[r,c]; out["m2"][m]=m2[r,c]
    ring=(m3*9-m1*1)/8; out["ring"][m]=ring[r,c]; out["prom"][m]=S["s1"][r,c]-ring[r,c]
    lm=ndi.maximum_filter(S["s1"],size=int(2*px(0.5))+1); out["lmax1"][m]=lm[r,c]
    hm=disc_mean(1.0)  # fraction of 1m disc above half of peak
    pk=S["s1"][r,c]; R=int(round(px(1.0)))
    for j,(a,b) in enumerate(zip(r,c)):
        w=S["s1"][max(0,a-R):a+R+1,max(0,b-R):b+R+1]; out["hmA"][m[j]]=(w>=0.5*pk[j]).mean()
    out["bg"][m]=np.median(prob[valid]); out["vfrac"][m]=valid.mean()
    print(s, len(m), "step",round(st,3), "s1 vs conf corr", np.round(np.corrcoef(out["s1"][m],df.conf[m])[0,1],3), flush=True)
out["conf"]=df.conf; out["src"]=df.src; out["reg"]=df.reg; out["grp"]=df.grp
pickle.dump(out,open(W+f"night/mf_{PROBS}.pkl","wb"))
