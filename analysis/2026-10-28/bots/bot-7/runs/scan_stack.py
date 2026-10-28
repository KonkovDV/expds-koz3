import ijson, sys, json
from collections import Counter
keys=Counter(); cls=Counter(); grp=Counter(); reg=set(); nv=[]; ex={}
with open('/workspace/rerank/stack_arch.geojson','rb') as f:
    for i,feat in enumerate(ijson.items(f,'features.item', use_float=True)):
        p=feat['properties']; keys.update(p.keys())
        k=(p.get('class_name'),p.get('threshold_group') or p.get('markup_type'))
        cls[k]+=1; reg.add(p.get('region_name'))
        nv.append(len(feat['geometry']['coordinates'][0]))
        if k not in ex: ex[k]=p
print(i+1); print(keys); 
for k,v in sorted(cls.items(), key=lambda x:-x[1]): print(k,v)
print(len(reg)); import statistics; print('verts median',statistics.median(nv), max(nv))
json.dump({str(k):v for k,v in ex.items()}, open('work/stack_examples.json','w'), ensure_ascii=False, indent=1)
