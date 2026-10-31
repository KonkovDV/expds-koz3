from ev import *
from shapely.geometry import shape
from scipy.spatial import cKDTree
class DF(dict):
    __getattr__=dict.__getitem__
    def __len__(s): return len(s['conf'])
    def xy(s): return np.c_[s['x'],s['y']]
def table(F):
    xy=np.array([mod._to_utm(shape(f["geometry"]),"auto").centroid.coords[0] for f in F])
    df=DF(x=xy[:,0],y=xy[:,1],conf=[f["properties"]["confidence"] for f in F],
        src=[f["properties"]["source"] for f in F],reg=[f["properties"]["region_name"] for f in F],grp=[metric.threshold_group(f["properties"]) for f in F])
    for k in df: df[k]=np.array(df[k])
    return df
def with_conf(F,conf,keep=None,grp=None):
    out=[]
    for i,f in enumerate(F):
        if keep is not None and not keep[i]: continue
        p={**f["properties"],"confidence":float(conf[i])}
        if grp is not None: p["threshold_group"]=grp[i]
        out.append({"type":"Feature","geometry":f["geometry"],"properties":p})
    return out
def nms(df,d,cross=True,same=True,soft=None):
    """greedy by conf; suppress neighbours within d (same region). soft: multiply conf by soft instead of drop"""
    conf=df.conf.copy(); keep=np.ones(len(df),bool); o=np.argsort(-conf,kind="stable")
    t=cKDTree(df.xy()); reg=df.reg; src=df.src
    nb=t.query_ball_point(df.xy(), d)
    if soft is None:
        for i in o:
            if not keep[i]: continue
            for j in nb[i]:
                if j==i or not keep[j] or reg[j]!=reg[i] or conf[j]>conf[i] or (conf[j]==conf[i] and j<i): continue
                if (src[j]==src[i] and same) or (src[j]!=src[i] and cross): keep[j]=False
        return conf,keep
    newc=conf.copy()
    for i in o:
        for j in nb[i]:
            if j==i or reg[j]!=reg[i] or conf[j]>conf[i] or (conf[j]==conf[i] and j<i): continue
            if (src[j]==src[i] and same) or (src[j]!=src[i] and cross): newc[j]*=soft
    return newc,keep
