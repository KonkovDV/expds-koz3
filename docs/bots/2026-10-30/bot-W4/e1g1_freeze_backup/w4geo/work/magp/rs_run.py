from rs import *
import sys
d,F,O=split(); df=table(F)
h,_=feat_hits({"type":"FeatureCollection","features":F},GTF); y=np.array(h,float)
cache="X.pkl"
if os.path.exists(cache): X,names=pickle.load(open(cache,"rb"))
else: X,names=build(F,df,y); pickle.dump((X,names),open(cache,"wb"))
fo=np.array([fold_of(r) for r in df.reg]); K=[k for k in FOLDS if (fo==k).any()]
print("folds",K, {k:int((fo==k).sum()) for k in K})
sel=[i for i,nm in enumerate(names) if nm in (sys.argv[1].split(",") if len(sys.argv)>1 and sys.argv[1]!="all" else names)]
Xs=X[:,sel]; nr=int(sys.argv[2]) if len(sys.argv)>2 else 300
# single-level OOF
s1=np.zeros(len(y))
for k in K: s1[fo==k],_=fit_pred(Xs,y,fo!=k,fo==k,nr)
from metric_nested import auc
print("AUC conf",round(auc(y,df.conf),4),"oof",round(auc(y,s1),4))
# nested honest: for held k, scores for others from models w/o {j,k}
from metric_nested import nested_eval
res=nested_eval(F,df,Xs,y,fo,K,nr)
pickle.dump(dict(s1=s1,res=res,names=[names[i] for i in sel]),open(f"rs_{len(sel)}_{nr}.pkl","wb"))
