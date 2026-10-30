import json,sys,glob,collections
# build_oof.py <name> <view: flipwbf_m300|flipwbf_m100> <comma sites>
name,view,sites=sys.argv[1],sys.argv[2],sys.argv[3].split(',')
base=json.load(open('/workspace/koz3/out/w1-dfine-ae-final/oof_dfine_ae_flipwbf.geojson'))
S=set(sites); keep=[f for f in base['features'] if f['properties']['region_name'] not in S]
add=[]
for s in sites:
    d=json.load(open(f'/workspace/koz3/w1/v8/out/{s}.json')); add+=d['features'][view]
c=collections.Counter(f['properties']['region_name'] for f in base['features'] if f['properties']['region_name'] in S)
print(name,'removed',sum(c.values()),dict(c),'added',len(add))
out=dict(base); out['features']=keep+add
json.dump(out,open(f'/workspace/koz3/w1/v8/oof/{name}.geojson','w'),ensure_ascii=False)
