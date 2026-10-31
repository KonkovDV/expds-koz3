"""exec A night: rescoring screen on E1 MagGr finds: conf'=f(conf, map feats) -> M3 rules (q.9 A.5, soft D2 X.7) -> honest finds F1 + MagGr P/R"""
import sys, pickle
sys.path.insert(0,"/workspace/koz3/w4geo/work/magp")
from feats import *
from scipy.stats import rankdata
def roc_auc_score(y,s):
    r=rankdata(s); y=np.asarray(y,bool); n1=y.sum(); n0=len(y)-n1
    return (r[y].sum()-n1*(n1+1)/2)/(n1*n0)
d,F,O=split(W+"cstar/C_CetO3_Sr8G_noTerFl_L_E1.geojson"); df=table(F)
mf=pickle.load(open(W+"night/mf_probs_n40r8.pkl","rb"))
h,_=feat_hits({"type":"FeatureCollection","features":F},GTF); hit=np.array(h,bool)
def m3(c,Q=.9,A=.5,D=2.0,X=.7):
    c=c.copy()
    if A<1:
        for s in np.unique(df.src):
            m=df.src==s; v=c[m]; c[m]=np.clip(0.9*v**A*(v/np.quantile(v,Q))**(1-A),0,1)
    c,_=nms(DF(**{**df,"conf":c}),D,1,1,soft=X); return c
def run(c,tag,**kw):
    cc=m3(np.clip(c,1e-6,1),**kw); o,_=score(with_conf(F,cc),tag); return o
if __name__=="__main__":
    conf=df.conf.astype(float); eps=1e-4
    feats={k:mf[k] for k in ["s0","s1","s2","s4","m1","m2","ring","prom","hmA","lmax1"]}
    feats["prom_r"]=mf["prom"]/(mf["s1"]+eps); feats["m1_r"]=mf["m1"]/(mf["s1"]+eps); feats["s4_r"]=mf["s4"]/(mf["s1"]+eps); feats["ring_r"]=mf["ring"]/(mf["s1"]+eps)
    print("AUC within raster (mean over rasters weighted), conf vs feats")
    for k,v in [("conf",conf)]+list(feats.items()):
        a=[];w=[]
        for s in np.unique(df.src):
            m=(df.src==s)&np.isfinite(v)
            if hit[m].sum()>5 and (~hit[m]).sum()>5: a.append(roc_auc_score(hit[m],v[m])); w.append(hit[m].sum())
        print(f"{k:8s} {np.average(a,weights=w):.4f}")
    run(conf,"E1_M3_repro")
