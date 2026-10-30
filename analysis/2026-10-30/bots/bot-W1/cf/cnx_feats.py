"""W5 x W2: lidar ConvNeXt c3-sc OOF (2 seeds, honest 5-fold, li_cnx) -> (a) reranker features for every stack
candidate, (b) seed-fused candidate set appended to the stack (V6c). See PREREG_w5_rerank.md (V5x, V6c)."""
from common import *
import numpy as np, shapely, pickle, argparse
from shapely.strtree import STRtree
from expds import metric as M
mod = M.load_official(METRIC); calc = mod.F1Calculator()
ap = argparse.ArgumentParser()
ap.add_argument('--oof', default='/workspace/koz3/w2/oof/merged/oof_cnx_c3-sc_s{}.geojson')
ap.add_argument('--merged_out', default='/workspace/koz3/w5data_cnx/stack_arch_cnx.geojson')
ap.add_argument('--skip_merge', action='store_true')
args = ap.parse_args()
KUR = {'kurgany_povrezhdennye', 'kurgany_tselye'}
fam = lambda c: 'kurgany' if c in KUR else c
S = []
for s in (0, 1):
    F = load(args.oof.format(s))['features']
    pr = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': F}, False); M._UTM_GEOM_CACHE.clear()
    it = {c: iter(v) for c, v in pr.items()}  # same per-class UTM projection as prep.py (official metric path)
    geoms = np.array([next(it[mod.normalize_class_name(f['properties']['class_name'])])[0] for f in F], dtype=object)
    S.append(dict(F=F, g=geoms, c=np.array([f['properties']['confidence'] for f in F]),
                  cl=np.array([f['properties']['class_name'] for f in F]), fam=np.array([fam(f['properties']['class_name']) for f in F]),
                  reg=np.array([f['properties']['region_name'] for f in F]), tree=STRtree(geoms)))
    print('seed', s, len(F), flush=True)
covered = set(S[0]['reg']) | set(S[1]['reg'])
# ---- (b) seed fusion: s0 anchored, s1 partner = max-conf intersecting det of same family
q0, q1 = S[1]['tree'].query(S[0]['g'], predicate='intersects')
ok = S[0]['fam'][q0] == S[1]['fam'][q1]; q0, q1 = q0[ok], q1[ok]
best1 = np.zeros(len(S[0]['c'])); np.maximum.at(best1, q0, S[1]['c'][q1])
used1 = np.zeros(len(S[1]['c']), bool); used1[q1] = True
fused = []
for i, f in enumerate(S[0]['F']):
    p = dict(f['properties']); p['confidence'] = round(float((S[0]['c'][i] + best1[i]) / 2), 6); p['threshold_group'] = 'li_cnx'
    p['fid'] = f'cnx0_{i}'; fused.append({'type': 'Feature', 'geometry': f['geometry'], 'properties': p})
for j in np.flatnonzero(~used1):
    f = S[1]['F'][j]; p = dict(f['properties']); p['confidence'] = round(float(S[1]['c'][j] / 2), 6); p['threshold_group'] = 'li_cnx'
    p['fid'] = f'cnx1_{j}'; fused.append({'type': 'Feature', 'geometry': f['geometry'], 'properties': p})
if not args.skip_merge:
  print('fused', len(fused), 'agree', int((best1 > 0).sum()), 's1-only', int((~used1).sum()), flush=True)
  with open(W / 'stack_arch.geojson', 'rb') as fh:
      fh.seek(0, 2); n = fh.tell(); fh.seek(n - 64); tail = fh.read()
  k = tail.rstrip().rfind(b']}'); assert k > 0
  cut = n - 64 + k
  with open(args.merged_out, 'wb') as out, open(W / 'stack_arch.geojson', 'rb') as fh:
      left = cut
      while left > 0:
          b = fh.read(min(1 << 24, left)); out.write(b); left -= len(b)
      for f in fused: out.write(b', ' + json.dumps(f, ensure_ascii=False).encode())
      out.write(b']}')
  print('merged ->', args.merged_out, flush=True)
del fused
for d in S: d.pop('F', None)
import gc; gc.collect()
# ---- (a) features for stack candidates
T = pickle.load(open(W / 'work/table.pkl', 'rb')); T = {k: T[k] for k in ('CL', 'cls', 'reg')}; gc.collect(); CL = T['CL']
geom = shapely.from_wkb(pickle.load(open(W / 'work/geom_wkb.pkl', 'rb')))
N = len(geom); cls = T['cls'].astype(int)
cname = np.array([CL[c] if c >= 0 else '' for c in cls]); cfam = np.array([fam(c) for c in cname])
valid = np.flatnonzero((cls >= 0) & ~shapely.is_missing(geom))
X = np.full((N, 5), np.nan, np.float32)
cov = np.isin(T['reg'], list(covered)); X[cov] = 0
for s in (0, 1):
    A_, B_ = [], []
    for c0 in range(0, len(valid), 200000):
        v = valid[c0:c0 + 200000]; a_, b_ = S[s]['tree'].query(geom[v], predicate='intersects'); A_.append(v[a_]); B_.append(b_)
    a = np.concatenate(A_); b = np.concatenate(B_); same_fam = cfam[a] == S[s]['fam'][b]; same = cname[a] == S[s]['cl'][b]
    m = np.zeros(N); np.maximum.at(m, a[same_fam], S[s]['c'][b][same_fam]); X[cov, s] = m[cov]
    m2 = np.zeros(N); np.maximum.at(m2, a[same], S[s]['c'][b][same]); X[cov, 4] = np.maximum(X[cov, 4], m2[cov])
    print('feat seed', s, int((m > 0).sum()), flush=True)
X[:, 2] = (X[:, 0] + X[:, 1]) / 2; X[:, 3] = np.where(np.isnan(X[:, 0]), np.nan, ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(np.float32))
X2 = np.load(W / 'work/X2.npy'); names = json.load(open(W / 'work/X2_names.json'))
np.save(W / 'work/X3.npy', np.c_[X2, X].astype(np.float32))
json.dump(names + ['cnx_s0', 'cnx_s1', 'cnx_mean', 'cnx_agree', 'cnx_samecls'], open(W / 'work/X3_names.json', 'w'))
print('X3 done', 'covered rows', int(cov.sum()), 'hit rows', int((X[:, 2] > 0).sum()), flush=True)
