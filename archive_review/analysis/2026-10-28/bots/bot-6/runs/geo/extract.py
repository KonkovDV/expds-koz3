"""Вытащить из кэша /workspace/rerank/work (read-only) только object_poly: кандидаты стека (UTM WKB, conf, участок, группа)
и эталон object_poly (UTM WKB, участок, markup). Выход: opoly.pkl."""
import sys, pickle, gc, resource
resource.setrlimit(resource.RLIMIT_AS, (3_800_000_000, 3_800_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np
from common import *
from expds import metric as M
T = pickle.load(open(W/'work/table.pkl', 'rb'))
CL = T['CL']; c = CL.index('object_poly')
idx = np.flatnonzero(T['cls'] == c)
out = dict(idx=idx, conf=T['conf'][idx], reg=T['reg'][idx].astype(str), grp=T['grp'][idx].astype(str), mk=T['mk'][idx].astype(str),
           area=T['area'][idx], n_gt=T['n_gt']['object_poly'], greg=list(T['greg']['object_poly']))
P, J, S = T['pairs'][('object_poly', 'object_poly')]; out['pairs_base'] = (P, J, S)
del T; gc.collect()
g = pickle.load(open(W/'work/geom_wkb.pkl', 'rb')); out['wkb'] = [g[i] for i in idx]; del g; gc.collect()
G = pickle.load(open(W/'work/gt_wkb.pkl', 'rb')); out['gt_wkb'] = list(G['object_poly']); del G; gc.collect()
mod = M.load_official(METRIC); calc = mod.F1Calculator()
gt = load(W/'gt_train_v2.geojson'); gt['features'] = [f for f in gt['features'] if mod.normalize_class_name(f['properties'].get('class_name', '')) == 'object_poly']
gts = M._collect(mod, calc, gt, True)['object_poly']
out['gmk'] = [x[4] if len(x) > 4 else x[3] for x in gts]
assert [x[2] for x in gts] == out['greg'] and len(gts) == out['n_gt']
pickle.dump(out, open('opoly.pkl', 'wb'), protocol=5)
from collections import Counter
print('cands', len(idx), Counter(zip(out['grp'], out['mk'])), 'gt', out['n_gt'], Counter(out['gmk']))
print('regions etgr/terfl', sorted(set(r for r, gg in zip(out['reg'], out['grp']) if gg in ('etgr', 'terfl'))))
print('gt regions by mk', {m: sorted(Counter(r for r, mm in zip(out['greg'], out['gmk']) if mm == m).items()) for m in set(out['gmk'])})
print('maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
