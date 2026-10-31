import json,sys,math
sys.path.insert(0,'/workspace/koz3_shared/bot-7/repo')
from shapely.geometry import shape,mapping,Point
from expds import metric as M, sp_dfine as S
MP='/workspace/koz3_shared/bot-7/repo/materials/compute_metrics.py'
R=('101_','103_')
gt=json.load(open('/workspace/rerank/gt_train_v2.geojson'))
gtf=[f for f in gt['features'] if f['properties'].get('region_name','')[:4] in R and f['properties'].get('markup_type')=='sp_or']
print('gt sp_or',len(gtf), gtf[0]['properties'])
G=dict(gt,features=gtf)
def sc(name,feats):
    feats=[f for f in feats if f['properties']['markup_type']=='sp_or']
    for i,f in enumerate(feats): f['properties']['fid']=i
    r=M.score(dict(type='FeatureCollection',crs=gt.get('crs'),features=feats),G,MP)
    pc={k:(round(v.get('f1',0),4) if isinstance(v,dict) else v) for k,v in r.get('per_class',{}).items()}
    print(f'{name:32s} n={len(feats):4d} final={r["final_score"]:.4f}', pc)
def micro(f,rad=0.5):
    if f['properties']['class_name'] in('kurgany_povrezhdennye','kurgany_tselye'):
        c=shape(f['geometry']).centroid
        f=dict(f,geometry=mapping(Point(c.x,c.y).buffer(rad/S.ground_scale(c.y),quad_segs=8)))
    return f
for v in('res_v2','res_v3'):
    sc(v,[f for f in json.load(open(f'{v}/result.geojson'))['features'] if f['properties']['region_name'][:4] in R])
# OOF stack (fold models), thresholds from config
T={'sp_or':{'gorodishcha':0.994085,'kurgany_povrezhdennye':0.3645,'kurgany_tselye':0.3725,'arkhitektury':0.286725},
   'sp_focal':{'kurgany_povrezhdennye':0.3645,'kurgany_tselye':0.3725},'sp_arch':{'arkhitektury':0.286725},
   'sp_focal2':{'kurgany_povrezhdennye':0.501645,'kurgany_tselye':0.645505}}
st=[json.loads(l) for l in open('stack_101_103.jsonl')]
st=[f for f in st if f['properties']['markup_type']=='sp_or']
keep=[]
for f in st:
    p=f['properties']; g=p.get('threshold_group') or 'sp_or'
    if p['confidence']>=T.get(g,{}).get(p['class_name'],9): keep.append(micro(f))
sc('OOF stack sp_* (A, без rerank)',keep)
sc('OOF stack sp_focal* только курганы',[f for f in keep if f['properties']['class_name'].startswith('kurg')])
TH={"kurgany_povrezhdennye": 0.555265, "kurgany_tselye": 0.670285, "gorodishcha": 0.68847,"arkhitektury": 0.373015, "fortifikatsii": 0.869051}
g=json.load(open('/workspace/kaggle_out/koz3-dfine-spor-full-rest/oof_dfine_f0.geojson'))
dets=[dict(geom=shape(f['geometry']),class_name=f['properties']['class_name'],confidence=f['properties']['confidence'],region_name=f['properties']['region_name'],model=0) for f in g['features']]
cl=[c for c in S.cluster_circles(dets) if c['confidence']>=TH[c['class_name']]]
ff=[dict(type='Feature',geometry=mapping(c['geom']),properties=dict(class_name=c['class_name'],confidence=c['confidence'],region_name=c['region_name'],markup_type='sp_or',crs='EPSG:3857')) for c in cl]
sc('OOF D-FINE f0 C r4',ff)
sc('OOF D-FINE f0 C micro0.5',[micro(f) for f in ff])
