from feats import *
d,F,O=split(); df=table(F); c0=df.conf
def per_src(c,fn):
    o=np.zeros(len(c))
    for s in np.unique(df.src):
        m=df.src==s; o[m]=fn(c[m])
    return np.clip(o,0,1)
def softc(c,dd,x,cross=1):
    dfx=DF(**{**df,"conf":c}); cs,_=nms(dfx,dd,cross,1,soft=x); return cs
_,pb=score(F,"E1",False)
def go(c,tag):
    o,per=score(with_conf(F,c),tag,False)
    fd={}
    for k,v in E.FOLDS.items():
        a=[per[r] for r in per if r[:3] in v]; b=[pb[r] for r in pb if r[:3] in v]
        if not a: continue
        f=lambda L:(2*sum(x[0] for x in L))/(2*sum(x[0] for x in L)+sum(x[1] for x in L)+sum(x[2]-x[0] for x in L)) if False else None
        tp=sum(x[0] for x in a); fp=sum(x[1] for x in a); g=sum(x[2] for x in a); tpb=sum(x[0] for x in b); fpb=sum(x[1] for x in b)
        F1=lambda tp,fp,g: 2*(tp/(tp+.5*fp))*(tp/g)/((tp/(tp+.5*fp))+(tp/g)) if tp else 0
        fd[k]=round(F1(tp,fp,g)-F1(tpb,fpb,g),4)
    print(json.dumps({**o,"dF1_fold":fd},ensure_ascii=False),flush=True)
for q,a in ((0.95,0.5),(0.95,0.0),(0.9,0.5),(0.9,0.0),(0.98,0.5)):
    cn=per_src(c0,lambda v: v**a*(v/np.quantile(v,q))**(1-a)*0.9)
    go(cn,f"norm q{q} a{a}")
    for dd,x in ((1.75,0.7),(2.0,0.7),(2.0,0.6)):
        go(softc(cn,dd,x),f"norm q{q} a{a} + soft{dd} x{x}")
for dd,x in ((1.5,0.5),(1.75,0.7),(2.0,0.7),(2.0,0.75),(2.0,0.65)):
    go(softc(c0,dd,x),f"soft{dd} x{x}")
