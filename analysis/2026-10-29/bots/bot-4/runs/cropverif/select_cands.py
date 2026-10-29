"""Выбор оптических кандидатов пула (не городища) и их центроиды в EPSG:3857 из stack_arch.geojson."""
import pickle, numpy as np, ijson, collections
from shapely.geometry import shape
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb')); pool = np.load('/workspace/rerank/work/pool.npy')
sel = pool & np.isin(T['cls'], [0, 1, 3, 4, 5, 6]) & np.isin(T['mk'], ['sp_or', 'ae', 'or'])
idx = np.flatnonzero(sel); want = set(idx.tolist())
print(len(idx), collections.Counter(zip(T['grp'][idx].tolist(), T['cls'][idx].tolist())))
cx = np.full(len(idx), np.nan); cy = np.full(len(idx), np.nan); pos = {v: k for k, v in enumerate(idx)}
with open('/workspace/rerank/stack_arch.geojson', 'rb') as fh:
    for i, f in enumerate(ijson.items(fh, 'features.item', use_float=True)):
        if i in want:
            p = f['properties']; k = pos[i]
            assert str(p.get('region_name')) == T['reg'][i] and str(p.get('markup_type')) == T['mk'][i], i
            c = shape(f['geometry']).centroid; cx[k] = c.x; cy[k] = c.y
    n = i + 1
assert n == len(T['cls']), (n, len(T['cls']))
np.savez('optical_pool_cands.npz', stack_idx=idx.astype(np.int32), region=T['reg'][idx], mk=T['mk'][idx], grp=T['grp'][idx],
         cls=T['cls'][idx], x3857=cx, y3857=cy)
print('stack features', n, 'nan', np.isnan(cx).sum())
