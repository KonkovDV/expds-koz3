# Статистика OOF koz3-cnx-li-c3sc-f23 + проверка формата: по участкам/классам/уверенности; GT по участкам (gt_train_v2, gt_li_cnx_*)
import json,collections,sys
sys.path.insert(0,'/workspace/koz3_shared/bot-10/repo')
from expds.naming import site_family
from shapely.geometry import shape
K='/workspace/kaggle_out/koz3-cnx-li-c3sc-f23/'
rep={}
fc=json.load(open(K+'oof_cnx_c3-sc.geojson')); oof=fc['features']
rep['fc_crs']=fc.get('crs'); rep['n_oof']=len(oof)
keys=collections.Counter(tuple(sorted(f['properties'])) for f in oof); rep['keys']={'|'.join(k):v for k,v in keys.items()}
for k in ('threshold_group','markup_type','crs','class_name'):
    rep[k]=dict(collections.Counter(f['properties'].get(k) for f in oof))
gt_=collections.Counter(f['geometry']['type'] for f in oof); rep['geom_type']=dict(gt_)
bad=0; xs=[];ys=[]; areas=[]
for f in oof:
    g=shape(f['geometry']); bad+= (not g.is_valid) or g.is_empty; c=g.centroid; xs.append(c.x); ys.append(c.y); areas.append(g.area)
rep['invalid_or_empty']=bad; rep['centroid_x_range']=[min(xs),max(xs)]; rep['centroid_y_range']=[min(ys),max(ys)]
areas.sort(); rep['area_m2_min_med_max']=[areas[0],areas[len(areas)//2],areas[-1]]
conf=sorted(f['properties']['confidence'] for f in oof); rep['conf_min_med_max']=[conf[0],conf[len(conf)//2],conf[-1]]
# участки внутри боксов bot-10 family_clusters.json
B=json.load(open('/workspace/koz3_shared/bot-10/work/family_clusters.json'))['boxes']; out=0
for f,x,y in zip(oof,xs,ys):
    b=B.get(f['properties']['region_name'])
    if b and not (b[0]-50<=x<=b[2]+50 and b[1]-50<=y<=b[3]+50): out+=1
rep['centroid_outside_region_box_50m']=out
rf=json.load(open(K+'region_fold.json'))['region_fold']
rep['fold_count']=dict(collections.Counter(rf.get(f['properties']['region_name']) for f in oof))
TH=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
by=collections.defaultdict(lambda: collections.Counter())
for f in oof:
    p=f['properties']; c=p['confidence']
    for t in TH:
        if c>=t: by[(p['region_name'],p['class_name'])][t]+=1
rep['det_by_region_class_thr']={f'{r}|{c}':{str(t):v[t] for t in TH} for (r,c),v in sorted(by.items())}
gt=json.load(open('/workspace/rerank/gt_train_v2.geojson'))['features']
fams={site_family(f['properties']['region_name']) for f in oof}
g=collections.Counter()
for f in gt:
    p=f['properties']
    if site_family(p['region_name']) in fams: g[(p['region_name'],p.get('markup_type'),p['class_name'])]+=1
rep['gt_v2_families']={'|'.join(map(str,k)):v for k,v in sorted(g.items())}
for fo in (2,3):
    G=json.load(open(K+f'gt_li_cnx_c3-sc_f{fo}.geojson'))['features']
    rep[f'gt_li_cnx_f{fo}']={'|'.join(k):v for k,v in collections.Counter((f['properties']['region_name'],f['properties']['class_name']) for f in G).items()}
json.dump(rep,open('stats.json','w'),ensure_ascii=False,indent=1)
print(json.dumps({k:v for k,v in rep.items() if k not in ('det_by_region_class_thr','gt_v2_families','gt_li_cnx_f2','gt_li_cnx_f3')},ensure_ascii=False))
