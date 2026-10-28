"""Симуляция: копия каждого объекта EtGr в каждый слой, чья рамка (extra/*_границы) содержит его точку.
Сопоставление как в layer_recall (центр в эталоне, жадно по уверенности, класс общий на участок)."""
import json, glob, collections, os, re
from shapely.geometry import shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree
from pyproj import Transformer
ROOT='/workspace/data/train-expds-3'
tr=Transformer.from_crs(32636,3857,always_xy=True).transform
gt=json.load(open('/workspace/rerank/gt_train_v2.geojson'))
G=collections.defaultdict(list)
for x in gt['features']:
    p=x['properties']
    if p['markup_type']=='etgr': G[p['region_name']].append(shape(x['geometry']))
P=collections.defaultdict(list)
for f in glob.glob('/workspace/koz3_shared/bot-10/work/fam/*.jsonl'):
    for l in open(f):
        if '"etgr"' not in l: continue
        x=json.loads(l); p=x['properties']
        if p.get('markup_type')=='etgr': P[p['region_name']].append((shape(x['geometry']).representative_point(),p['confidence']))
L={}
for r in G:
    ex=os.path.join(ROOT,r,'extra'); lay=[]
    for b in sorted(glob.glob(ex+'/*EtGr*_границы.geojson')):
        d=json.load(open(b))
        u=unary_union([shape(ft['geometry']) for ft in d['features']])
        lay.append(transform(tr,u))
    L[r]=lay
def run(thr, copy, cap=99):
    tp=fp=0
    for r,g in G.items():
        tree=STRtree(g); used=set(); out=[]
        for pt,c in P[r]:
            if c<thr: continue
            n=min(cap,max(1,sum(1 for b in L[r] if b.contains(pt)))) if copy else 1
            out+= [(c,pt)]*n
        for c,pt in sorted(out,key=lambda t:-t[0]):
            hit=[int(i) for i in tree.query(pt) if int(i) not in used and g[int(i)].contains(pt)]
            if hit: used.add(hit[0]); tp+=1
            else: fp+=1
    return tp,fp
# остальное object_poly из stack_arch_base: maggr 146/78, terfl 319/38; GT 1083
def f1(tp,fp,gt=1083):
    tp+=78+38; fp+=(146-78)+(319-38)
    P_=tp/(tp+0.5*fp) if tp+fp else 0; R=tp/gt; return round(2*P_*R/(P_+R),4) if P_+R else 0
res={}
for thr in []:
    a=run(thr,False); b=run(thr,True)
    res[thr]=dict(base=a,base_f1=f1(*a),copy=b,copy_f1=f1(*b))
    print(thr,res[thr],flush=True)
print({r:len(v) for r,v in L.items()})
json.dump({str(k):v for k,v in res.items()},open('etgr_layer_sim.json','w'),indent=1)
res2={}
for cap in [1,2,3,4]:
    for thr in [0.0,0.1]:
        b=run(thr,True,cap); res2[f'cap{cap}_thr{thr}']=dict(copy=b,f1=f1(*b)); print(cap,thr,b,f1(*b),flush=True)
json.dump(res2,open('etgr_layer_sim_cap.json','w'),indent=1)
