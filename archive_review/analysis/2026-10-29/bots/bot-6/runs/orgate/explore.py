import sys; sys.path.insert(0,'/workspace/rerank')
from common import *
import fast, numpy as np
from collections import Counter
T=fast.T; cls=T['cls'].astype(int); C=fast.cid['arkhitektury']; a=np.flatnonzero(cls==C)
pool=np.load(W/'work/pool.npy'); y=np.load(W/'work/first.npy')
print('keys', list(T.keys()))
print('arch cands', a.size, 'pool', pool[a].sum(), 'first-hit', y[a].sum())
print('groups', Counter(T['grp'][a]).most_common())
print('mk', Counter(T['mk'][a]).most_common() if 'mk' in T else '')
fam=Counter(site_family(r) for r in T['reg'][a]); print('families', len(fam), fam.most_common(15))
print('regions', len(set(T['reg'][a])))
gf=Counter(site_family(r) for r in T['greg']['arkhitektury']); print('GT arch', sum(gf.values()), 'families', len(gf), gf.most_common(15))
pf=Counter(site_family(r) for r in T['reg'][a[y[a]]]); print('first-hit families', pf.most_common())
print('conf q', np.quantile(T['conf'][a],[0,.5,.9,.99,1]))
print('pairs arch', {k:len(v[0]) for k,v in T['pairs'].items() if 'arkhitektury' in k})
