import json,sys,time,cProfile,pstats
sys.path.insert(0,'/workspace/koz3/repo')
import os; os.environ.setdefault('EXPDS_NO_SHM','1')
from expds import metric
gt=json.load(open('/workspace/koz3/ds/gt/gt_train_v2.geojson'))
pred=json.load(open(sys.argv[1]))
print('n_pred',len(pred['features']))
t=time.time()
pr=cProfile.Profile(); pr.enable()
c=metric.build_curves(pred,gt)
pr.disable()
print('build_curves',time.time()-t)
pstats.Stats(pr).sort_stats('cumulative').print_stats(25)
