"""exec A night 30->31: per-raster in-sample optimal cut vs GT-free raster stats (diagnostic only)"""
from feats import *
P=sys.argv[1] if len(sys.argv)>1 else W+'cstar/C_CetO3_Sr8G_noTerFl_L_E1_M3.geojson'
d,F,O=split(P); df=table(F)
h,c=feat_hits({"type":"FeatureCollection","features":F},GTF)
hit=np.array(h,bool)
from scipy.spatial import ConvexHull
gtr=c.gt_regions
for s in np.unique(df.src):
    m=df.src==s; cf=df.conf[m]; ht=hit[m]; o=np.argsort(-cf); cum=np.cumsum(ht[o]); n=np.arange(1,len(o)+1)
    g=gtr[df.reg[m][0]]
    Pm=cum/(cum+.5*(n-cum)); R=cum/g; F1=2*Pm*R/(Pm+R+1e-9); k=F1.argmax()
    try: A=ConvexHull(df.xy()[m]).volume
    except: A=np.nan
    q=np.quantile(cf,[.5,.75,.9,.95,.99])
    print(f"{s[:42]:42s} n={m.sum():5d} gt={g:4d} hits={ht.sum():4d} optk={k+1:5d} optcut={cf[o][k]:.3f} F1={F1[k]:.3f} R={R[k]:.3f} | A={A:7.0f} dens={m.sum()/A*100:6.2f} n>.5={int((cf>.5).sum()):4d} q={np.round(q,3)}")
