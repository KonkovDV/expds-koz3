from common import *
import fast
t=time.time(); cv = fast.curves(); print('fast curves', time.time()-t)
gt = load(W/'gt_train_v2.geojson'); new, optical = sets(gt); del gt
print('insample', insample(cv)[0])
s, counts, info, folds = honest(cv, new, optical); print('honest', s, info)
