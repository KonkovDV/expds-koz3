# Статистика OOF koz3-cnx-li-c3sc-f01: по участкам/классам/уверенности; GT по участкам (gt_train_v2, gt_li_cnx_*)
import json,collections,sys
sys.path.insert(0,'/workspace/koz3_shared/bot-10/repo')
from expds.naming import site_family
K='/workspace/kaggle_out/koz3-cnx-li-c3sc-f01/'
rep={}
oof=json.load(open(K+'oof_cnx_c3-sc.geojson'))['features']
rf=json.load(open(K+'region_fold.json'))
fold_of=rf if isinstance(next(iter(rf.values())),int) else None
rep['n_oof']=len(oof)
keys=collections.Counter(tuple(sorted(f['properties'])) for f in oof); rep['keys']={'|'.join(k):v for k,v in keys.items()}
tg=collections.Counter(f['properties'].get('threshold_group') for f in oof); rep['threshold_group']=dict(tg)
TH=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
by=collections.defaultdict(lambda: collections.Counter())
for f in oof:
    p=f['properties']; c=p['confidence']
    for t in TH:
        if c>=t: by[(p['region_name'],p['class_name'])][t]+=1
rep['det_by_region_class_thr']={f'{r}|{c}':{str(t):v[t] for t in TH} for (r,c),v in sorted(by.items())}
gt=json.load(open('/workspace/rerank/gt_train_v2.geojson'))['features']
regs=sorted({f['properties']['region_name'] for f in oof})
g=collections.Counter()
for f in gt:
    p=f['properties']
    if p['region_name'] in regs or site_family(p['region_name']) in ('Гнёздово','Тимерево'):
        g[(p['region_name'],p.get('markup_type'),p['class_name'])]+=1
rep['gt_v2_families']={'|'.join(map(str,k)):v for k,v in sorted(g.items())}
for fo in (0,1):
    G=json.load(open(K+f'gt_li_cnx_c3-sc_f{fo}.geojson'))['features']
    rep[f'gt_li_cnx_f{fo}']={'|'.join(k):v for k,v in collections.Counter((f['properties']['region_name'],f['properties']['class_name']) for f in G).items()}
json.dump(rep,open('stats.json','w'),ensure_ascii=False,indent=1)
print(json.dumps({k:v for k,v in rep.items()},ensure_ascii=False,indent=0))
