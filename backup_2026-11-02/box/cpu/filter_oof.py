import ijson, json, sys
src, dst, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
out=[]; n=0; keys=None
with open(src,'rb') as f:
    for feat in ijson.items(f,'features.item', use_float=True):
        n+=1; p=feat['properties']
        if keys is None: keys=list(p); print('props', p, flush=True)
        c=p.get('confidence', p.get('score',1.0))
        if c>=thr: out.append(feat)
        if n%200000==0: print(n,len(out),flush=True)
json.dump({"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::3857"}},"features":out}, open(dst,'w'))
print('done',n,len(out))
