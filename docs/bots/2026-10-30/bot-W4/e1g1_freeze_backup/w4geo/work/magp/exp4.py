from feats import *
import pickle
d,F,O=split(); df=table(F); c0=df.conf
X,names=pickle.load(open("X.pkl","rb")); res=X[:,names.index("res")]; n=X[:,names.index("n")]
cv=np.array(["цвет" in s for s in df.src])
def softc(c,dd,x):
    cs,_=nms(DF(**{**df,"conf":c}),dd,1,1,soft=x); return cs
G={}
G["L"]=df.grp
G["n500"]=np.array([("cv" if a else "gr")+("lo" if b<500 else "hi") for a,b in zip(cv,n)]); G["n500"]=np.array(["maggr_pix_"+g for g in G["n500"]])
G["coarse_lo"]=np.array([("maggr_pix_cvlo" if (a and (b<300 or r>0.15)) else g) for a,b,r,g in zip(cv,n,res,df.grp)])
G["coarse_all_lo"]=np.array([(("maggr_pix_cvlo" if a else "maggr_pix_grlo") if r>0.15 else g) for a,r,g in zip(cv,res,df.grp)])
G["fine_cv2_sep"]=np.array([("maggr_pix_cv2" if "цвет_2" in s else g) for s,g in zip(df.src,df.grp)])
G["cvhi_split_res"]=np.array([("maggr_pix_cvhic" if (g=="maggr_pix_cvhi" and r>0.05) else g) for r,g in zip(res,df.grp)])
for nm,g in G.items():
    for sd in (None,(2.0,0.7),(1.75,0.7)):
        c=c0 if sd is None else softc(c0,*sd)
        o,per=score(with_conf(F,c,grp=g),f"{nm} soft{sd}",False)
        print(json.dumps(o,ensure_ascii=False), {r[:3]:round(v[0]/v[2],3) for r,v in per.items() if r[:3] in ("200","207","209","210","211","212")},flush=True)
