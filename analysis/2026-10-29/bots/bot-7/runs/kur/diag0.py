"""bot-7 task 7, диагностика 0: курганы на спутнике без лидара — нет кандидата vs есть (по table.pkl стека)."""
import pickle,json,collections,sys,numpy as np
sys.path.insert(0,'/workspace/expds-koz3')
from expds.naming import site_family
T=pickle.load(open('/workspace/rerank/work/table.pkl','rb'))
boxes=set(json.load(open('/workspace/expds-koz3/notes/las_boxes_59.json')))
gt=json.load(open('/workspace/rerank/gt_train_v2.geojson'))
out={}
for C in ('kurgany_povrezhdennye','kurgany_tselye'):
    greg=np.array(T['greg'][C]); n=len(greg)
    P,J,S=T['pairs'][(C,C)]; cov=np.zeros(n,bool); cov[np.unique(J)]=True
    # markup_type of GT in same order as greg (as in common.load_gt for gor)
    mk=[f['properties'].get('markup_type') for f in gt['features'] if (f['properties'].get('class_name') or '')==C and f.get('geometry')]
    mk=np.array(mk) if len(mk)==n else None
    print(C,n,'mk aligned',mk is not None)
    if mk is None: continue
    for typ in ('sp_or','ae'):
        sel=(mk==typ)&np.array([r not in boxes for r in greg])
        fam=collections.defaultdict(lambda:[0,0])
        for j in np.where(sel)[0]: f=site_family(greg[j]); fam[f][0]+=1; fam[f][1]+=int(cov[j])
        out[f'{C}|{typ}|nolidar']=dict(gt=int(sel.sum()),covered=int(cov[sel].sum()),by_family=dict(sorted(fam.items(),key=lambda x:-x[1][0])))
        print(typ,'nolidar GT',sel.sum(),'covered',cov[sel].sum(),dict(list(sorted(fam.items(),key=lambda x:-x[1][0]))[:10]))
json.dump(out,open('diag0.json','w'),ensure_ascii=False,indent=1)
