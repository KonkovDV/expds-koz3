from feats import *
d,F,O=split(); df=table(F); c0=df.conf
def per_src(c,fn):
    o=np.zeros(len(c))
    for s in np.unique(df.src):
        m=df.src==s; o[m]=fn(c[m])
    return np.clip(o,0,1)
def softc(c,dd,x):
    cs,_=nms(DF(**{**df,"conf":c}),dd,1,1,soft=x); return cs
rows=[]
for q in (0.85,0.9,0.95):
  for a in (0.25,0.5,0.75,1.0):
    cn=c0 if a==1.0 else per_src(c0,lambda v: v**a*(v/np.quantile(v,q))**(1-a)*0.9)
    if a==1.0 and q!=0.9: continue
    for dd in (1.75,2.0,2.25):
      for x in (0.6,0.7,0.8):
        o,_=score(with_conf(F,softc(cn,dd,x)),f"q{q} a{a} d{dd} x{x}",False); rows.append(o)
        print(o["tag"],o["finds_F1"],o["mag_P"],o["mag_R"],flush=True)
json.dump(rows,open("exp7.json","w"))
