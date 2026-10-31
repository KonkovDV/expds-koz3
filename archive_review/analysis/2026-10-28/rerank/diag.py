from common import *
import fast, numpy as np, sys
from sklearn.metrics import roc_auc_score
T=fast.T; cls=T['cls'].astype(int); conf=T['conf']; grp=T['grp']; CL=fast.CL; N=len(cls)
z=np.load(W/f'work/nested_{sys.argv[1]}.npz'); outer=z['outer']
okc=np.load(W/'work/okc.npy'); y=okc[np.arange(N),np.maximum(cls,0)]&(cls>=0)
S=conf.copy(); m=cls>=0; S[m]=outer[m]
rb=sweep_groups(fast.curves()); rm=sweep_groups(fast.curves(score=S))
for c in CL:
    print(f"{c:22s} base F1 {rb[c]['f1']:.4f} model-OOF F1 {rm[c]['f1']:.4f}")
    for g in rb[c]['groups']:
        ii=(cls==fast.cid[c])&(grp==g)
        if y[ii].sum()>0 and (~y[ii]).sum()>0:
            a0=roc_auc_score(y[ii],conf[ii]); a1=roc_auc_score(y[ii],S[ii])
        else: a0=a1=float('nan')
        b=rb[c]['groups'][g]; mm=rm[c]['groups'].get(g,{})
        print(f"   {g:10s} n={ii.sum():7d} pos={y[ii].sum():5d} AUC {a0:.3f}->{a1:.3f}  kept/tp {b['kept']}/{b['tp']} -> {mm.get('kept')}/{mm.get('tp')}")
