import ijson, numpy as np, json
from collections import defaultdict
c=defaultdict(list)
with open('/workspace/rerank/stack_arch.geojson','rb') as f:
    for p in ijson.items(f,'features.item.properties', use_float=True):
        c[(p['class_name'], p.get('threshold_group') or p.get('markup_type'))].append(p['confidence'])
out={}
for k,v in c.items():
    v=np.sort(np.array(v))[::-1]
    out['|'.join(k)]=v.tolist()
np.save('work/confs.npy', out, allow_pickle=True)
base=json.load(open('/workspace/rerank/stack_arch_base.json'))['per_class']
for k,v in sorted(out.items()):
    cl,g=k.split('|'); v=np.array(v)
    t=base.get(cl,{}).get('groups',{}).get(g,{}).get('threshold')
    print(f"{k:40s} n={len(v):7d} thr={t} q: top1000={v[min(999,len(v)-1)]:.4f} top20000={v[min(19999,len(v)-1)]:.4f} n>=thr/2={int((v>=(t or 0)/2).sum()) if t else '-'}")
