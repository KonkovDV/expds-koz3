"""Проверка гипотезы bot-2: li-объекты стека A (work/fam) из другого OOF, чем дамп li_hard2."""
import json,glob,os,numpy as np,collections
from shapely.geometry import shape
from scipy.spatial import cKDTree
z=np.load('/workspace/koz3_shared/bot-2/work/lireal/dump_compact.npz')
regs=list(z['regions']); cls=list(z['classes'])
allc=collections.Counter(); dconf=[]
R={r:i for i,r in enumerate(regs)}; C={c:i for i,c in enumerate(cls)}; trees={}
def tree(ri,ci):
    k=(ri,ci)
    if k not in trees:
        m=(z['region']==ri)&(z['cls']==ci); trees[k]=(cKDTree(np.c_[z['x'][m],z['y'][m]]) if m.any() else None, z['conf'][m])
    return trees[k]
for fp in sorted(glob.glob('work/fam/*.jsonl')):
    for ln in open(fp,encoding='utf-8'):
        if '"li"' not in ln: continue
        f=json.loads(ln); p=f['properties']
        if p.get('markup_type')!='li' or p.get('threshold_group') is not None or p['region_name'] not in R or p['class_name'] not in C: continue
        cn=p['class_name']; T,cf=tree(R[p['region_name']],C[cn]); allc[(cn,'n')]+=1
        if T is None: continue
        c=shape(f['geometry']).centroid; d,j=T.query([c.x,c.y])
        if d<0.5:
            allc[(cn,'match')]+=1; dd=abs(float(p['confidence'])-cf[j]); dconf.append(dd); allc[(cn,'conf_eq_1e-3')]+= dd<1e-3
out={' '.join(k):v for k,v in sorted(allc.items())}
dc=np.array(dconf); out['dconf_median']=float(np.median(dc)); out['dconf_p90']=float(np.quantile(dc,.9)); out['dconf_max']=float(dc.max())
print(json.dumps(out,ensure_ascii=False,indent=1,default=int))
