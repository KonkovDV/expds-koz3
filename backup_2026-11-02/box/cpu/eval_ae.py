import json, sys, time, importlib.util
from multiprocessing import Pool
sys.path.insert(0,'/workspace/koz3/ds/e2e/score')
spec=importlib.util.spec_from_file_location('cm','/workspace/koz3/ds/e2e/score/compute_metrics.py'); cm=importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
gt=json.load(open('/workspace/koz3/ds/gt/gt_train_v2.geojson'))
G=[f for f in gt['features'] if f['properties']['markup_type']=='ae']
def fc(x): return {"type":"FeatureCollection","features":x}
P={}
def load(name):
    global P
    P[name]=[f for f in json.load(open(f'/workspace/koz3/cpu/{name}_oof_ge02.geojson'))['features'] if f['properties']['confidence']>=0.3]
    for f in P[name]: f['properties']={k:f['properties'][k] for k in ('class_name','confidence','markup_type','region_name')}
def run(args):
    name, taus = args
    pr=[f for f in P[name] if f['properties']['confidence']>=taus.get(f['properties']['class_name'],1.01)]
    t=time.time(); r=cm.F1Calculator().calculate_f1(fc(pr),fc(G))
    cls={k:{kk:v.get(kk) for kk in ('precision','recall','f1')} for k,v in r.get('class_results',r.get('per_class',{})).items()} if isinstance(r.get('class_results',r.get('per_class')),dict) else None
    return dict(name=name,taus=taus,n=len(pr),final=r['final_score'],sec=round(time.time()-t),cls=cls)
if __name__=='__main__':
    load(sys.argv[1])
    C=['kurgany_tselye','kurgany_povrezhdennye','gorodishcha','fortifikatsii','arkhitektury']
    jobs=[]
    oof={'kurgany_tselye':0.45,'kurgany_povrezhdennye':0.5,'gorodishcha':1.01,'fortifikatsii':0.35,'arkhitektury':1.01}
    for n in P:
        jobs.append((n,oof))
        for t in (0.3,0.4,0.5,0.6,0.7,0.8):
            jobs.append((n,{c:t for c in C}))
            jobs.append((n,{**{c:t for c in C},'gorodishcha':1.01,'arkhitektury':1.01}))
    with Pool(int(sys.argv[2])) as pool:
        for r in pool.imap_unordered(run,jobs):
            print(json.dumps(r,ensure_ascii=False),flush=True)
