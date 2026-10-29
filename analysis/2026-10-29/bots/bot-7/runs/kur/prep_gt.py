"""bot-7 task7: эталон курганов (povr, tselye) в UTM с регионом и markup -> gt_kur.pkl. Порядок = gt_wkb.pkl = индексы J в table.pkl."""
import sys,json,pickle,numpy as np,shapely
sys.path.insert(0,'/workspace/expds-koz3'); from expds import metric as M
mod=M.load_official('/workspace/expds-koz3/materials/compute_metrics.py')
CL=['kurgany_povrezhdennye','kurgany_tselye']
gt=json.load(open('/workspace/rerank/gt_train_v2.geojson'))
mk={c:[] for c in CL}; rg={c:[] for c in CL}
for f in gt['features']:
    p=f.get('properties') or {}; c=mod.normalize_class_name(p.get('class_name','unknown'))
    if c in mk and f.get('geometry'): mk[c].append(p.get('markup_type')); rg[c].append(p.get('region_name'))
gw=pickle.load(open('/workspace/rerank/work/gt_wkb.pkl','rb'))
T=pickle.load(open('/workspace/rerank/work/table.pkl','rb'))
out={}
for c in CL:
    assert len(gw[c])==len(mk[c])==T['n_gt'][c]
    greg=np.asarray(T['greg'][c]).astype(str)
    print(c,'region agree',np.mean(greg==np.asarray(rg[c]).astype(str)))
    out[c]=dict(wkb=gw[c],mk=np.array(mk[c]).astype(str),reg=greg)
    g=shapely.from_wkb(gw[c]); print(c,'bounds sample',g[0].bounds, 'area p10/50/90',np.percentile(shapely.area(g),[10,50,90]))
pickle.dump(out,open('gt_kur.pkl','wb'))
