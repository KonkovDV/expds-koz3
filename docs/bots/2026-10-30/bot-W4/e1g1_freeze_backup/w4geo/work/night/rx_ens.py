from rx import *
conf=df.conf.astype(float); eps=1e-3
S={p:np.load(W+f"night/s1_{p}.npy") for p in ["probs_n40r8","probs_n40","probs_n40r6","probs_n40r10","probs_n15","probs_aug","probs_gw1d","probs_gw3d","probs"]}
base=S["probs_n40r8"]
def auc(v):
    a=[];w=[]
    for s in np.unique(df.src):
        m=df.src==s
        if hit[m].sum()>5 and (~hit[m]).sum()>5: a.append(roc_auc_score(hit[m],v[m])); w.append(hit[m].sum())
    return np.average(a,weights=w)
for p,v in S.items(): print(p, round(auc(v),4))
sets={"r8+r6+r10":["probs_n40r8","probs_n40r6","probs_n40r10"],"r8+r10":["probs_n40r8","probs_n40r10"],
      "r8+aug":["probs_n40r8","probs_aug"],"n40fam":["probs_n40r8","probs_n40r6","probs_n40r10","probs_n40"],
      "all8":[p for p in S if p!="probs"],"r8+r6+r10+aug":["probs_n40r8","probs_n40r6","probs_n40r10","probs_aug"]}
for k,ps in sets.items():
    e=np.mean([S[p] for p in ps],0); c=np.clip(conf*(e+eps)/(base+eps),0,1)
    print(k,"AUC",round(auc(c),4)); run(c,"ens_"+k)
