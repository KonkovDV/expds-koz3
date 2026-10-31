"""MagGr finds OOF rescoring (nested by geo5 folds) + honest eval. GT used only as training labels on other folds."""
from feats import *
import glob, rasterio, lightgbm as lgb, os, pickle
T="/workspace/koz3/w4geo/train/"
def raster_info(src,reg):
    p=glob.glob(T+reg+"/**/"+src,recursive=True)[0]
    with rasterio.open(p) as ds: return abs(ds.res[0]) if ds.crs and ds.crs.is_projected else np.nan, ds.count
def build(F,df,hits):
    n=len(df); X={}
    c=df.conf; X["conf"]=c
    srcs=np.unique(df.src); X["typ_cv"]=np.array(["цвет" in s for s in df.src],float); X["cv2"]=np.array(["цвет_2" in s for s in df.src],float)
    for k in ("q50","q90","q99","n","rank","rel90","z"): X[k]=np.zeros(n)
    X["res"]=np.zeros(n); X["nb"]=np.zeros(n)
    for s in srcs:
        m=df.src==s; v=c[m]; reg=df.reg[m][0]
        X["q50"][m]=np.median(v); X["q90"][m]=np.quantile(v,.9); X["q99"][m]=np.quantile(v,.99); X["n"][m]=m.sum()
        X["rank"][m]=(np.argsort(np.argsort(-v))+1)/m.sum(); X["rel90"][m]=v/(np.quantile(v,.9)+1e-6); X["z"][m]=(v-v.mean())/(v.std()+1e-6)
        X["res"][m],X["nb"][m]=raster_info(s,reg)
    X["nrast"]=np.array([len(set(df.src[df.reg==r])) for r in df.reg],float) if False else np.zeros(n)
    rr={r:len(set(df.src[df.reg==r])) for r in set(df.reg)}; X["nrast"]=np.array([rr[r] for r in df.reg],float)
    t=cKDTree(df.xy()); xy=df.xy()
    for d in (1.0,2.0,3.0,5.0,10.0):
        nb=t.query_ball_point(xy,d); same=np.zeros(n); cross=np.zeros(n); mxs=np.zeros(n); mxc=np.zeros(n)
        for i,js in enumerate(nb):
            for j in js:
                if j==i or df.reg[j]!=df.reg[i]: continue
                if df.src[j]==df.src[i]: same[i]+=1; mxs[i]=max(mxs[i],c[j])
                else: cross[i]+=1; mxc[i]=max(mxc[i],c[j])
        X[f"ns{d}"]=same; X[f"nx{d}"]=cross
        if d in (2.0,5.0): X[f"mxs{d}"]=mxs; X[f"mxc{d}"]=mxc; X[f"dom{d}"]=c-mxs
    area=np.array([mod._to_utm(shape(f["geometry"]),"auto").area for f in F]); X["area"]=area
    names=list(X); return np.c_[tuple(X[k] for k in names)], names
FOLDS=E.FOLDS
def fold_of(r): return [k for k,v in FOLDS.items() if r.split("_")[0] in v][0]
PAR=dict(objective="binary",learning_rate=0.05,num_leaves=15,min_data_in_leaf=40,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,verbose=-1,num_threads=3,seed=0)
def fit_pred(X,y,tr,te,nr=300):
    m=lgb.train(PAR,lgb.Dataset(X[tr],y[tr]),nr); return m.predict(X[te]), m
