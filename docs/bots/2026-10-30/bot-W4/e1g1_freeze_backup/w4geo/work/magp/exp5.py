from feats import *
import pickle
d,F,O=split(); df=table(F); c0=df.conf
X,names=pickle.load(open("X.pkl","rb")); res=X[:,names.index("res")]; n=X[:,names.index("n")]
cv=np.array(["цвет" in s for s in df.src])
def softc(c,dd,x):
    cs,_=nms(DF(**{**df,"conf":c}),dd,1,1,soft=x); return cs
G={}
G["R1_cv1_gr_n300"]=np.array(["maggr_pix_cv" if a else g for a,g in zip(cv,df.grp)])
G["R2_type"]=np.array(["maggr_pix_cv" if a else "maggr_pix_gr" for a in cv])
G["R5_cvlo_coarse_and_212"]=np.array([("maggr_pix_cvlo" if (a and (b<300 or r>0.15 or "цвет_2" in s)) else g) for a,b,r,g,s in zip(cv,n,res,df.grp,df.src)])
G["R6_cv_n1000"]=np.array([("maggr_pix_cvlo" if (a and b<1000) else g) for a,b,g in zip(cv,n,df.grp)])
for nm,g in G.items():
    for sd in (None,(2.0,0.7)):
        c=c0 if sd is None else softc(c0,*sd)
        o,per=score(with_conf(F,c,grp=g),f"{nm} soft{sd}",False)
        print(json.dumps(o,ensure_ascii=False), {r[:3]:round(v[0]/v[2],3) for r,v in per.items() if r[:3] in ("200","207","209","210","211","212")},flush=True)
