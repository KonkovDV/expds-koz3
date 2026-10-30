from rx import *
import itertools, json as J
conf=df.conf.astype(float); eps=1e-3
P=["probs_n40r8","probs_n40","probs_n40r6","probs_n40r10","probs_n15","probs_aug","probs_gw1d","probs_gw3d"]
S={p:np.load(W+f"night/s1_{p}.npy") for p in P}; base=S["probs_n40r8"]
sets={"r8r6r10aug":["probs_n40r8","probs_n40r6","probs_n40r10","probs_aug"],"all8":P,"r8aug":["probs_n40r8","probs_aug"],
 "r8r6r10aug_gw":["probs_n40r8","probs_n40r6","probs_n40r10","probs_aug","probs_gw1d","probs_gw3d"],"r8aug_gw3":["probs_n40r8","probs_aug","probs_gw3d"]}
res=[]
for k,ps in sets.items():
    e=np.mean([S[p] for p in ps],0); c0=np.clip(conf*(e+eps)/(base+eps),0,1)
    for A,X,D in itertools.product([.25,.5,.75,1.0],[.6,.7,.8,1.0],[1.75,2.0]):
        if X==1.0 and D==2.0: continue
        o=run(c0,f"{k}_A{A}_X{X}_D{D}",A=A,X=X,D=D); res.append(o)
J.dump(res,open("rx_ens2.json","w"))
ok=[o for o in res if o["mag_P"]>=0.6705]
for o in sorted(ok,key=lambda o:-o["mag_R"])[:15]: print("OK",o)
for o in sorted(res,key=lambda o:-o["finds_F1"])[:10]: print("F1",o)
