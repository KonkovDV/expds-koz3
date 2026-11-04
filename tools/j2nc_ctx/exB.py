"""Production-like context features: stack rows (V3P drop3) + C19/D-FINE extras, all at production early floors (output_floors)."""
import os, sys, json, pickle, time, tomllib
import numpy as np
os.chdir('/workspace/koz3/w4geo/x_v4on'); sys.path.insert(0, '.')
from expds import w5cf_infer as W, pipeline as PL
import shapely.wkb, shapely.geometry
D = '/workspace/koz3/w1/cf/w5d/work/'; OUT = '/workspace/koz3/orch/bot-W1/work/j2nc/'
cfg = tomllib.load(open('configs/default.toml', 'rb'))
T = pickle.load(open(D + 'table.pkl', 'rb')); G = pickle.load(open(D + 'geom_wkb.pkl', 'rb'))
X4 = np.load(D + 'X4.npy', mmap_mode='r'); names = json.load(open(D + 'X4_names.json'))
pool = np.load(D + 'pool.npy')
E = pickle.load(open(OUT + 'extras.pkl', 'rb')); dreg = set(E['dreg']); SP = set(E['SP'])
CL = T['CL']; cls = np.asarray(T['cls']); conf = np.asarray(T['conf'], float); reg = np.asarray(T['reg']).astype(str)
grp = np.asarray(T['grp']).astype(str); mk = np.asarray(T['mk']).astype(str)
FLc = {}
def fl(g, c):
    if g not in FLc: FLc[g] = PL.output_floors(cfg, g)
    return FLc[g].get(c, 0.15)
drop3 = np.array([g == 'etgr' or (g in SP and r in dreg) for g, r in zip(grp, reg)])
flo = np.array([fl(g, CL[c]) if c >= 0 else np.inf for g, c in zip(grp, cls)])
keep = (~drop3) & (conf >= flo)
score_rows = keep & pool & (mk == 'li')
exr = {}
for f in E['ex']:
    if f['confidence'] >= fl(f['threshold_group'], f['class_name']): exr.setdefault(f['region_name'], []).append(f)
json.dump({g: v for g, v in FLc.items()}, open(OUT + 'ctx_floors.json', 'w'), indent=1, ensure_ascii=False)
gqi = names.index('gq'); cfi = [i for i, n in enumerate(names) if n.startswith('cf_')]
Xp = np.full((len(conf), len(names)), np.nan, np.float32)
regs = sorted(set(reg[score_rows])); print('regions', len(regs), 'score rows', int(score_rows.sum()),
      'extras kept', sum(len(v) for v in exr.values()), flush=True)
g0, p0 = W.gq_of, W.in_pool; art = W.load_artifacts(cfg); t0 = time.time()
for n_, rn in enumerate(regs):
    ii = np.flatnonzero((reg == rn) & keep)
    feats = []
    for j in ii:
        f = {'class_name': CL[cls[j]], 'confidence': float(conf[j]), 'markup_type': mk[j], 'threshold_group': grp[j],
             'region_name': rn, 'geom': shapely.wkb.loads(G[j])}
        v = np.asarray(X4[j, cfi], np.float32)
        if mk[j] == 'li' and np.isfinite(v).any(): f[W.KEY] = v
        feats.append(f)
    for e in exr.get(rn, []):
        feats.append({k: e[k] for k in ('class_name', 'confidence', 'markup_type', 'threshold_group', 'region_name')} |
                     {'geom': shapely.geometry.shape(e['geometry'])})
    ns = len(ii)
    W.reset()
    if X4[ii[0], names.index('has_las')] == 1: W._STATE['las_regions'].add(rn)
    tb = W.table(feats, art)
    assert (tb['rows'][:ns] == np.arange(ns)).all() if len(tb['rows']) >= ns else False, rn
    qtr = np.array(X4[ii, gqi])
    def gq_patch(g, c, cf_, a_, _q=qtr, _ns=ns):
        q = g0(g, c, cf_, a_); q[:_ns] = _q; return q
    W.gq_of = gq_patch
    try:
        X = W.features(tb, art)
    finally:
        W.gq_of = g0
    sr = score_rows[ii]
    Xp[ii[sr]] = X[:ns][sr]
    if n_ % 10 == 0: print(n_, rn, len(feats), f'{time.time()-t0:.0f}s', flush=True)
np.save(OUT + 'Xp.npy', Xp); np.savez(OUT + 'ctx_masks.npz', keep=keep, score_rows=score_rows, drop3=drop3)
print('done', f'{time.time()-t0:.0f}s', int(np.isfinite(Xp[:, 0]).sum()))
