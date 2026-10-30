"""screen E1-like file with new maggr finds: M3-style rules grid -> honest finds F1 + MagGr P/R"""
import sys,itertools
sys.path.insert(0,"/workspace/koz3/w4geo/work/magp")
from feats import *
P=sys.argv[1]; d,F,O=split(P); df=table(F); conf=df.conf.astype(float)
def m3(c,Q=.9,A=.5,D=2.0,X=.7):
    c=c.copy()
    if A<1:
        for s in np.unique(df.src):
            m=df.src==s; v=c[m]; c[m]=np.clip(0.9*v**A*(v/np.quantile(v,Q))**(1-A),0,1)
    if X<1: c,_=nms(DF(**{**df,"conf":c}),D,1,1,soft=X)
    return c
tag=P.split("/")[-1][:-8]; score(F,tag+"_raw")
for A,X,D in itertools.product([.25,.5,1.0],[.6,.7,1.0],[1.75,2.0]):
    if X==1.0 and D==2.0: continue
    score(with_conf(F,m3(conf,A=A,X=X,D=D)),f"{tag}_A{A}_X{X}_D{D}")
