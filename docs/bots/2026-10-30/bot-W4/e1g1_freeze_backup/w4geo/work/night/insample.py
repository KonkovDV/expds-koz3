"""in-sample (all-fold) group thresholds vs honest: is the REF gap honest-vs-insample?"""
from rx import *
def ins(c,tag):
    pred={"type":"FeatureCollection","features":with_conf(F,c)}
    cur=metric.build_curves(pred,GTF)["finds_points"]
    res=metric.sweep_groups({"finds_points":cur})["finds_points"]
    taus={g:v["threshold"] for g,v in res["groups"].items()}
    tau=np.array([taus.get(g,np.inf) for g in cur.pred_group]); k=cur.conf>=tau
    tp=cur.hit[k].sum(); fp=k.sum()-tp; P=tp/(tp+.5*fp); R=tp/cur.n_gt
    mtp=tp+MP["tp"]; mfp=fp+MP["n"]-MP["tp"]
    print(tag,"INSAMPLE n",k.sum(),"finds P/R",round(P,4),round(R,4),"F1",round(2*P*R/(P+R),4),"mag P/R",round(mtp/(mtp+.5*mfp),4),round(mtp/3726,4),{g:round(t,3) for g,t in taus.items()})
conf=df.conf.astype(float)
ins(m3(conf),"E1_M3")
P_=["probs_n40r8","probs_n40r6","probs_n40r10","probs_aug","probs_gw1d","probs_gw3d"]
e=np.mean([np.load(W+f"night/s1_{p}.npy") for p in P_],0); b=np.load(W+"night/s1_probs_n40r8.npy")
c0=np.clip(conf*(e+1e-3)/(b+1e-3),0,1); ins(m3(c0,A=.25,X=.7,D=1.75),"ens6_A.25_X.7_D1.75")
run(c0,"ens6_A.25_X.7_D1.75_honest",A=.25,X=.7,D=1.75)
