import sys; sys.path.insert(0, '/workspace/rerank')
from common import *
import fast, expds.metric as M
cv = fast.curves()
regions = {r for c in cv.values() for r in c.pred_region} | {r for c in cv.values() for r in c.gt_regions}
print('metric from', M.__file__, 'regions', len(regions))
for pol in ('drop', 'keep'):
    c, info = M.honest_counts(cv, M.family_folds(regions, site_family, 10), unseen=pol)
    print(pol, score_counts(c)[0], info)
