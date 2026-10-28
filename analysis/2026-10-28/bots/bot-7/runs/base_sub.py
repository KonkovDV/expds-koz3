from common7 import *
import resource
st=load(W/sys.argv[1]); gt=load('/workspace/rerank/gt_train_v2.geojson')
new,opt=sets(gt)
cur=build_curves(st,gt,METRIC)
ins,res=insample(cur); print('insample',ins)
s,counts,info,folds=honest(cur,new,opt); print('honest',s,info)
for c,v in score_counts(counts)[1].items(): print(c,{k:round(x,4) for k,x in v.items()})
pickle.dump({'counts':counts,'scores':s,'ins':ins},open(W/(sys.argv[1]+'.base.pkl'),'wb'))
print('maxrss MB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
