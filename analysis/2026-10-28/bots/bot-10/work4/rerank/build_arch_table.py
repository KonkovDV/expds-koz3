"""bot-10: таблица пар (кандидат arch, эталон arch, качество) для произвольного посемейного стека (fid = индекс в table.pkl).
Используется для микрокругов bot-7 (fam_arch) + проверка на исходном стеке (work/fam) против fast.PL."""
import sys, json, pickle, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/rerank')
from common import *
from expds import metric as M
from shapely.strtree import STRtree
import shapely
ap = argparse.ArgumentParser(); ap.add_argument('--fam'); ap.add_argument('--out'); a = ap.parse_args()
mod = M.load_official(METRIC); calc = mod.F1Calculator()
M._SHM_CACHE_DIR = Path('/nonexistent_ro/x')
T = pickle.load(open(W/'work/table.pkl', 'rb'))
A = 'arkhitektury'; C = T['CL'].index(A)
feats, fids = [], []
for f in sorted(Path(a.fam).glob('*.jsonl')):
    for line in open(f, encoding='utf-8'):
        if '"arkhitektury"' not in line: continue
        d = json.loads(line); p = d['properties']
        if mod.normalize_class_name(p.get('class_name')) != A: continue
        fids.append(int(p['fid'])); feats.append(d)
# fid в стеке — не индекс table.pkl: сопоставляем по региону и порядку (стек → семейные файлы сохраняет порядок)
from collections import defaultdict
ti = defaultdict(list)
for i in np.flatnonzero(T['cls'] == C): ti[T['reg'][i]].append(i)
pos = defaultdict(int); m = []
for d in feats:
    r = str(d['properties'].get('region_name', 'default_source')); m.append(ti[r][pos[r]]); pos[r] += 1
assert all(pos[r] == len(v) for r, v in ti.items()), 'count mismatch'
fids = np.array(m)
assert np.allclose(T['conf'][fids], [mod._get_confidence(f) for f in feats]), 'fid mismatch conf'
print('arch feats', len(fids), 'T arch', int((T['cls'] == C).sum()), flush=True)
pr = M._collect(mod, calc, {'type': 'FeatureCollection', 'features': feats}, False)[A]
geom = [x[0] for x in pr]; assert len(geom) == len(fids)
gg = [shapely.from_wkb(b) for b in pickle.load(open(W/'work/gt_wkb.pkl', 'rb'))[A]]
tree = STRtree([x.buffer(calc.r_tolerance) if x.geom_type == 'Point' else x for x in gg])
qp, qg = tree.query(geom)
P, J, S = [], [], []
for i, j in zip(qp, qg):
    pg = geom[i]
    if pg.geom_type != 'Polygon': continue
    ok, s = calc._match_quality(pg, gg[j])
    if ok: P.append(fids[i]); J.append(j); S.append(s)
P, J, S = np.array(P, np.int64), np.array(J, np.int64), np.array(S)
print('pairs', len(P), flush=True)
pickle.dump((P, J, S), open(a.out, 'wb'))
ref = T['pairs'][(A, A)]
k = lambda X: set(zip(X[0].tolist(), X[1].tolist()))
print('vs table.pkl orig pairs: same', len(k((P, J)) & k(ref)), 'only_new', len(k((P, J)) - k(ref)), 'only_orig', len(k(ref) - k((P, J))), flush=True)
