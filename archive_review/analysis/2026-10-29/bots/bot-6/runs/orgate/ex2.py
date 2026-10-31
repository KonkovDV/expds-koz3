import sys; sys.path.insert(0,'/workspace/rerank')
from common import *
import fast, numpy as np
from collections import Counter
T=fast.T; print(Counter(zip(T['mk'],T['grp'])).most_common(40))
import json; boxes=set(json.loads((ROOT/'notes'/'las_boxes_59.json').read_text(encoding='utf-8')))
C=fast.cid['arkhitektury']; a=np.flatnonzero(T['cls']==C); y=np.load(W/'work/first.npy')
las=np.array([r in boxes for r in T['reg'][a]]); print('arch in LAS regions', las.sum(), 'first-hit there', y[a][las].sum())
