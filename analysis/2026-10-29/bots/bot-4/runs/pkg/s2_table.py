"""Шаг 2: table.pkl пакета bot-7 (без twins, без crop) в формате /workspace/rerank/work/table.pkl.
Строки 0..N-1 = строки исходной table.pkl (тот же порядок → OOF реранкера применим как есть):
  курганы — геометрия из bot-7/work2/fam_kur (круги 0,5 м), пары матча пересчитаны официальным _match_quality;
  object_poly/finds_points старые — cls=-1 (выключены); новые строки stack6 (geo_pkg.jsonl) дописаны в конец."""
import sys, json, pickle, time
sys.path.insert(0, '/workspace/rerank')
from common import *
import numpy as np, shapely
from shapely.strtree import STRtree
from expds import metric as M
M._SHM_CACHE_DIR = Path('/nonexistent_ro/x')
mod = M.load_official(METRIC); calc = mod.F1Calculator()
t0 = time.time()
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}; N = len(T['cls'])
GW = pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))
G = {c: list(shapely.from_wkb(GW[c])) for c in CL}
assert all(len(G[c]) == T['n_gt'][c] for c in CL)
TREE = {c: STRtree([x.buffer(calc.r_tolerance) if x.geom_type == 'Point' else x for x in G[c]]) for c in CL if G[c]}
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
fam = np.array([site_family(r) for r in T['reg']])
def collect(feats):
    fc = {'type': 'FeatureCollection', 'features': feats}
    pr = M._collect(mod, calc, fc, False); M._UTM_GEOM_CACHE.clear()
    it = {c: iter(v) for c, v in pr.items()}; out = []
    for f in feats:
        c = mod.normalize_class_name(f['properties'].get('class_name', 'unknown')); out.append(next(it[c])[0])
    for v in it.values(): assert next(v, None) is None
    return out
def match(pc, idx, geoms, targets):
    res = {}
    for gc_ in targets:
        P, J, S = [], [], []
        if len(idx) and G[gc_]:
            qp, qg = TREE[gc_].query(geoms)
            for a, b in zip(qp, qg):
                pg = geoms[a]
                if pg.geom_type != 'Polygon': continue
                ok, s = calc._match_quality(pg, G[gc_][b])
                if ok: P.append(idx[a]); J.append(b); S.append(s)
        res[(pc, gc_)] = (P, J, S)
    return res
# --- курганы
newp = {(a, b): ([], [], []) for a in KUR for b in KUR}
area, perim, cx, cy, nv, bw, bh = (T[k].copy() for k in ['area', 'perim', 'cx', 'cy', 'nv', 'bw', 'bh'])
iskur = np.isin(T['cls'], [cid[c] for c in KUR])
FAM = Path('/workspace/koz3_shared/bot-7/work2/fam_kur')
nk = 0
for src in sorted(FAM.glob('*.jsonl')):
    F = src.stem
    old = np.flatnonzero(iskur & (fam == F))
    feats = []
    for line in open(src, encoding='utf-8'):
        if '"kurgany_' not in line: continue
        f = json.loads(line)
        if f['properties'].get('class_name') in KUR: feats.append(f)
    assert len(feats) == len(old), (F, len(feats), len(old))
    if not len(old): continue
    c_new = np.array([cid[f['properties']['class_name']] for f in feats]); r_new = np.array([f['properties']['region_name'] for f in feats])
    cf_new = np.array([mod._get_confidence(f) for f in feats])
    assert (c_new == T['cls'][old]).all() and (r_new == T['reg'][old]).all() and np.allclose(cf_new, T['conf'][old]), F
    geoms = []
    for i in range(0, len(feats), 50000): geoms += collect(feats[i:i + 50000])
    del feats
    geoms = np.array(geoms, dtype=object)
    for pc in KUR:
        m = c_new == cid[pc]
        for k, (P, J, S) in match(pc, old[m], list(geoms[m]), KUR).items():
            newp[k][0].extend(P); newp[k][1].extend(J); newp[k][2].extend(S)
    cen = shapely.centroid(geoms); b = shapely.bounds(geoms)
    area[old] = shapely.area(geoms); perim[old] = shapely.length(geoms); cx[old] = shapely.get_x(cen); cy[old] = shapely.get_y(cen)
    nv[old] = shapely.get_num_coordinates(geoms); bw[old] = b[:, 2] - b[:, 0]; bh[old] = b[:, 3] - b[:, 1]
    nk += len(old); print(F, len(old), nk, f'{time.time()-t0:.0f}s', flush=True)
assert nk == iskur.sum()
# --- geo (stack6)
feats = [json.loads(l) for l in open('geo_pkg.jsonl', encoding='utf-8')]
gg = np.array(collect(feats), dtype=object)
gcls = np.array([cid[mod.normalize_class_name(f['properties']['class_name'])] for f in feats], np.int16)
gconf = np.array([mod._get_confidence(f) for f in feats]); greg = np.array([str(f['properties'].get('region_name', 'default_source')) for f in feats])
ggrp = np.array([M.threshold_group(f['properties']) for f in feats]); gmk = np.array([str(f['properties'].get('markup_type') or '') for f in feats])
cls = T['cls'].copy(); cls[np.isin(cls, [cid['object_poly'], cid['finds_points']])] = -1
pairs = {k: v for k, v in T['pairs'].items() if k[0] not in KUR + ['object_poly', 'finds_points']}
for k, (P, J, S) in newp.items(): pairs[k] = (np.array(P, np.int64), np.array(J, np.int64), np.array(S))
for pc in ['object_poly', 'finds_points']:
    m = np.flatnonzero(gcls == cid[pc])
    for k, (P, J, S) in match(pc, N + m, list(gg[m]), [pc]).items(): pairs[k] = (np.array(P, np.int64), np.array(J, np.int64), np.array(S))
for k, v in pairs.items(): print(k, len(v[0]), 'old', len(T['pairs'].get(k, [[]])[0]))
cen = shapely.centroid(gg); b = shapely.bounds(gg)
out = dict(T)
out.update(cls=np.concatenate([cls, gcls]).astype(np.int16), conf=np.concatenate([T['conf'], gconf]), reg=np.concatenate([T['reg'], greg]),
           grp=np.concatenate([T['grp'], ggrp]), mk=np.concatenate([T['mk'], gmk]), area=np.concatenate([area, shapely.area(gg)]),
           perim=np.concatenate([perim, shapely.length(gg)]), cx=np.concatenate([cx, shapely.get_x(cen)]), cy=np.concatenate([cy, shapely.get_y(cen)]),
           nv=np.concatenate([nv, shapely.get_num_coordinates(gg)]), bw=np.concatenate([bw, b[:, 2] - b[:, 0]]), bh=np.concatenate([bh, b[:, 3] - b[:, 1]]),
           pairs=pairs, n_old=N)
Path('work').mkdir(exist_ok=True)
pickle.dump(out, open('work/table.pkl', 'wb'), protocol=5)
print('done', len(out['cls']), f'{time.time()-t0:.0f}s')
