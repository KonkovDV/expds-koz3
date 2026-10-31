from feats import *
d,F,O=split(); df=table(F)
c0=df.conf; out={}
def per_src(fn):
    c=np.zeros(len(c0))
    for s in np.unique(df.src):
        m=df.src==s; c[m]=fn(c0[m])
    return np.clip(c,0,1)
def go(c,tag,base=None):
    out[tag]=score(with_conf(F,c),tag)
go(c0,"E1")
for q in (0.95,0.99,1.0):
    for a in (0.0,0.5):
        go(per_src(lambda v: v**a*(v/np.quantile(v,q))**(1-a)*0.9), f"norm q{q} a{a}")
go(per_src(lambda v: 1-(np.argsort(np.argsort(-v))+0.5)/len(v)), "pct")
go(per_src(lambda v: 0.5*v+0.5*(1-(np.argsort(np.argsort(-v))+0.5)/len(v))), "0.5conf+0.5pct")
# soft-nms on top of norm
cs,_=nms(df,2.0,1,1,soft=0.7); go(cs,"soft2 x.7")
for d2,s2 in ((1.75,0.6),(2.0,0.6),(2.0,0.75),(2.25,0.7),(2.5,0.8),(2.0,0.65)):
    cs,_=nms(df,d2,1,1,soft=s2); go(cs,f"soft{d2} x{s2}")
    cs,_=nms(df,d2,0,1,soft=s2); go(cs,f"soft{d2} x{s2} sameonly")
