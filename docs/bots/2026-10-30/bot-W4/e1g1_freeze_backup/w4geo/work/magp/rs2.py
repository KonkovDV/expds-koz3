from rs import *
from metric_nested import auc
import pickle
d,F,O=split(); df=table(F)
h,_=feat_hits({"type":"FeatureCollection","features":F},GTF); y=np.array(h,float)
X,names=pickle.load(open("X.pkl","rb")); fo=np.array([fold_of(r) for r in df.reg]); K=["F1","F2","F3","F4"]
print(names)
def run(feats,nr,init=False,mono=True,lr=0.05):
    sel=[names.index(f) for f in feats]; Xs=X[:,sel]; s=np.zeros(len(y))
    par={**PAR,"learning_rate":lr}
    if mono: par["monotone_constraints"]=[1 if f=="conf" else 0 for f in feats]
    lc=np.log(np.clip(df.conf,1e-4,1-1e-4)/(1-np.clip(df.conf,1e-4,1-1e-4)))
    for k in K:
        tr,te=fo!=k,fo==k
        ds=lgb.Dataset(Xs[tr],y[tr],init_score=lc[tr] if init else None)
        m=lgb.train(par,ds,nr); s[te]=m.predict(Xs[te],raw_score=init)+(lc[te] if init else 0)
    pf={k:round(auc(y[fo==k],s[fo==k])-auc(y[fo==k],df.conf[fo==k]),4) for k in K}
    print(f"{','.join(feats)[:70]:70s} nr={nr} init={init} AUC {auc(y,s):.4f} (conf {auc(y,df.conf):.4f}) dAUC/fold {pf}",flush=True)
    return s
base=["conf"]
for fs in (["conf"],["conf","dom2.0","ns2.0","mxs2.0"],["conf","dom2.0","ns2.0","mxs2.0","nx2.0","mxc2.0"],["conf","rank","rel90","z"],["conf","dom2.0","ns2.0","mxs2.0","rank","rel90"],["conf","dom2.0","ns1.0","ns2.0","ns3.0","ns5.0","mxs2.0","dom5.0","mxs5.0"],["conf","area"]):
    run(fs,200); run(fs,100,init=True,lr=0.03)
