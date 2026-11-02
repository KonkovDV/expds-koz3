import json,sys,time
sys.path.insert(0,'/workspace/koz3/repo')
from expds import metric
from expds.naming import site_family
T=time.time
t=T(); gt=json.load(open('/workspace/koz3/ds/gt/gt_train_v2.geojson')); pred=json.load(open(sys.argv[1])); print('load',T()-t,len(pred['features']),flush=True)
t=T(); cur=metric.build_curves(pred,gt); print('build_curves',T()-t,flush=True)
t=T(); r=metric.sweep_groups(cur); print('sweep_groups',T()-t,flush=True)
regs=set();[regs.update(c.gt_regions) or regs.update(c.pred_region) for c in cur.values()]
t=T(); folds=metric.family_folds(sorted(regs),site_family,10); hc=metric.honest_counts(cur,folds); print('honest_counts',T()-t,flush=True)
t=T(); s=metric.score_counts(hc); print('score',s if not isinstance(s,tuple) else s[0],T()-t,flush=True)
t=T(); b=metric.paired_bootstrap_counts(hc,hc,site_family); print('paired_boot',T()-t,flush=True)
