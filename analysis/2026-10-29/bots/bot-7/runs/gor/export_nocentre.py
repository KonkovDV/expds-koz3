"""bot-7, task 6 addendum: варианты без центра сцены.
nocentre: все gen=='centre' удалены. centre_small: центр оставлен только в мелких сценах SpOr/Ae/Or (файл < 15 МиБ по S3-описи трейна;
на тесте то же правило по размеру файла). Прочие кандидаты те же (cand_id сохранены)."""
import json
sz={}
for ln in open('/workspace/data/s3_listing.txt',encoding='utf-8'):
    p=ln.split()
    if len(p)>=4 and p[2].isdigit() and p[3].endswith('.tif'): sz[(p[3].split('/')[0],p[3].split('/')[-1])]=int(p[2])/2**20
R=json.load(open('cands_gor_oof_D50.json')); Gj=json.load(open('cands_gor_oof_D50.geojson'))
small=lambda reg,f: sz.get((reg,f),1e9)<15
keep={'nocentre':lambda r: r['gen']!='centre','centre_small':lambda r: r['gen']!='centre' or small(r['reg'],r['file'])}
for name,fn in keep.items():
    A=[r for r in R if fn(r)]; ids={r['cand_id'] for r in A}
    json.dump(A,open(f'cands_gor_oof_D50_{name}.json','w'),ensure_ascii=False)
    F=[f for f in Gj['features'] if f['properties']['cand_id'] in ids]
    json.dump(dict(type='FeatureCollection',crs=Gj['crs'],features=F),open(f'cands_gor_oof_D50_{name}.geojson','w'),ensure_ascii=False)
    print(name,len(A),len(F),'centre kept',sum(r['gen']=='centre' for r in A))
