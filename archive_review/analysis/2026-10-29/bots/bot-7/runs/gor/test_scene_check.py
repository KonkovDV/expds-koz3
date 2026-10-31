"""bot-7, task 6 addendum: обрезаны ли тестовые сцены SpOr вокруг объекта.
Тест на box есть только описью организаторов (data_structure.md, размеры файлов). Сравниваем с трейном."""
import json, re, sys, collections, numpy as np
from shapely.geometry import Point
sys.path.insert(0,'/workspace/koz3_shared/bot-7/work/gor')
from common import load_gt, load_scenes
DS='/workspace/koz3_shared/competitors/yakirill99-ds-expedition/vendor/organizers/data_structure.md'
# --- test
test=collections.defaultdict(list); cur=None
for ln in open(DS,encoding='utf-8'):
    m=re.search(r'(\d+_region_\w+?_SpOr_[^ ]+\.tif) \(([\d.]+) (MB|GB|KB)\)',ln)
    if m:
        v=float(m.group(2))*{'MB':1,'GB':1024,'KB':1/1024}[m.group(3)]
        reg=re.search(r'region_(\w+?)_SpOr',m.group(1)).group(1); test[reg].append((m.group(1),v))
tregs_all=set(re.findall(r'── (region_\w+)\n',open(DS,encoding='utf-8').read()))
tmods=collections.Counter(re.findall(r'_(La|Ae|SpOr|Or|MagGr|EtGr|TerFl)_',open(DS,encoding='utf-8').read()))
# --- train sizes from S3 listing (bytes)
sz={}
for ln in open('/workspace/data/s3_listing.txt',encoding='utf-8'):
    p=ln.split()
    if len(p)>=4 and p[2].isdigit() and '_SpOr_' in p[3] and p[3].endswith('.tif'):
        sz[(p[3].split('/')[0],p[3].split('/')[-1])]=int(p[2])/2**20
S=load_scenes(); T,G,greg,gmk,lid,mod=load_gt()
cen=[g.centroid for g in G]
tr=[]
for s in S:
    if s['kind']!='SpOr': continue
    P=s['poly']; c=P.centroid; xs=[q[0] for q in s['corners']]; ys=[q[1] for q in s['corners']]
    W=max(xs)-min(xs); H=max(ys)-min(ys); half=0.5*np.hypot(W,H)
    js=[j for j in range(len(G)) if greg[j]==s['reg'] and P.contains(cen[j])]
    offs=[cen[j].distance(c) for j in js]
    tr.append(dict(reg=s['reg'],file=s['file'],mb=sz.get((s['reg'],s['file'])),ha=P.area/1e4,aspect=max(W,H)/max(1e-9,min(W,H)),
                   lidar=s['lidar'],w=s['w'],h=s['h'],gsd=s['gsd'],n_gor=len(js),off_m=min(offs) if offs else None,
                   off_rel=min(offs)/half if offs else None))
nfile=collections.Counter(r['reg'] for r in tr)
for r in tr: r['nfiles_reg']=nfile[r['reg']]
def pct(a,q=(10,25,50,75,90)): a=np.array([x for x in a if x is not None]); return [round(float(np.percentile(a,x)),1) for x in q] if len(a) else []
out={}
tm=[v for L in test.values() for _,v in L]
out['test']=dict(regions_total=len(tregs_all),mod_files=dict(tmods),spor_regions=len(test),spor_files=len(tm),
   mb_pct=pct(tm),n_lt15=sum(v<15 for v in tm),n_15_150=sum(15<=v<150 for v in tm),n_ge150=sum(v>=150 for v in tm),
   regions_multi=sum(len(L)>1 for L in test.values()),files=sorted(round(v,1) for v in tm),
   multi={k:[round(v,1) for _,v in L] for k,L in test.items() if len(L)>1})
trm=[r['mb'] for r in tr]
out['train']=dict(spor_regions=len(nfile),spor_files=len(tr),mb_missing=sum(m is None for m in trm),mb_pct=pct(trm),
   n_lt15=sum(m is not None and m<15 for m in trm),n_15_150=sum(m is not None and 15<=m<150 for m in trm),n_ge150=sum(m is not None and m>=150 for m in trm),
   regions_multi=sum(v>1 for v in nfile.values()))
# по корзинам размера: площадь, доля сцен с городищем, смещение
bins=[(0,15),(15,150),(150,1e9)]
out['train_bins']={}
for lo,hi in bins:
    R=[r for r in tr if r['mb'] is not None and lo<=r['mb']<hi]
    g=[r for r in R if r['n_gor']>0]
    out['train_bins'][f'{lo}-{hi}MB']=dict(n=len(R),ha_pct=pct([r['ha'] for r in R]),aspect_pct=pct([r['aspect'] for r in R]),
       gsd_pct=pct([r['gsd'] for r in R],(10,50,90)),with_gor=len(g),gor_off_m_pct=pct([r['off_m'] for r in g]),
       gor_off_rel_pct=pct([r['off_rel'] for r in g]),n_off_lt100m=sum(r['off_m']<100 for r in g),
       nonlidar=sum(not r['lidar'] for r in R),multi_file_reg=sum(r['nfiles_reg']>1 for r in R))
# mb ~ area: оценка площади теста по связи МБ->га на трейне (одна и та же GSD?)
a=np.array([(r['mb'],r['ha']) for r in tr if r['mb']]);
k=np.median(a[:,1]/a[:,0]); out['train_ha_per_mb_median']=round(float(k),3)
out['train_ha_per_mb_pct']=pct(a[:,1]/a[:,0])
out['test_ha_est_pct']=[round(x*k,1) for x in out['test']['mb_pct']]
json.dump(dict(summary=out,train_rows=tr),open('testchk/test_scene_check.json','w'),ensure_ascii=False,indent=1,default=str)
print(json.dumps(out,ensure_ascii=False,indent=1))
