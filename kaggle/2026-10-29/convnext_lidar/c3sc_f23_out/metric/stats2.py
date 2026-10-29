# Сводка по участкам + проверка дублей «один центр — два класса курганов»
import json,collections
K='/workspace/kaggle_out/koz3-cnx-li-c3sc-f23/'
S=json.load(open('stats.json'))
oof=json.load(open(K+'oof_cnx_c3-sc.geojson'))['features']
gt={}
for fo in (2,3):
    for k,v in S[f'gt_li_cnx_f{fo}'].items(): gt[k]=v
rows=[]
for k,v in S['det_by_region_class_thr'].items():
    rows.append((k,v['0.05'],v['0.3'],v['0.5'],gt.get(k,0)))
for r in rows: print(*r,sep='\t')
# дубли центров между povr/tselye
pos=collections.defaultdict(dict)
for f in oof:
    p=f['properties']; xs=[c[0] for c in f['geometry']['coordinates'][0]]; ys=[c[1] for c in f['geometry']['coordinates'][0]]
    key=(p['region_name'],round(sum(xs)/len(xs),1),round(sum(ys)/len(ys),1))
    pos[key][p['class_name']]=p['confidence']
both=[v for v in pos.values() if 'kurgany_povrezhdennye' in v and 'kurgany_tselye' in v]
nk=sum(1 for v in pos.values() if 'kurgany_povrezhdennye' in v or 'kurgany_tselye' in v)
print('kurgan centres',nk,'with both classes',len(both))
import statistics
print('example',both[:3])
json.dump({'rows':rows,'kurgan_centres':nk,'both_classes_same_centre':len(both)},open('stats2.json','w'),ensure_ascii=False,indent=1)
