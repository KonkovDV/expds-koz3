"""Оценка пилота Or-архитектуры на 008_Герите официальным _match_quality (compute_metrics.py).
Порог 0.45 задан заранее (из COMPETITORS_DEEP.md п.6), не подбирается. Кривая по порогам — только справочно."""
import sys, json, ast, pickle, collections
import numpy as np, shapely
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
from pathlib import Path
ROOT = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(ROOT))
from expds import metric as M
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
G = 'arkhitektury'; R = sys.argv[1] if len(sys.argv) > 1 else '008_Герите'; PFX = sys.argv[2] if len(sys.argv) > 2 else 'ger'
GG = shapely.from_wkb(pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))[G]); greg = np.array(T['greg'][G])
J = np.flatnonzero(greg == R); print('GT arch in', R, len(J), 'bounds', np.round(shapely.total_bounds(GG[J]), 1))
out = {}
def score(polys, confs, thr):
    o = np.argsort(-np.asarray(confs)); used = set(); tp = fp = 0
    for i in o:
        if confs[i] < thr: continue
        best, bs = -1, -1
        for j in J:
            if j in used: continue
            g = GG[j]; gg = g.buffer(calc.r_tolerance) if g.geom_type == 'Point' else g
            if not polys[i].intersects(gg): continue
            ok, s = calc._match_quality(polys[i], g)
            if ok and s > bs: best, bs = j, s
        if best >= 0: used.add(best); tp += 1
        else: fp += 1
    fn = len(J) - tp
    f1 = 2 * tp / (2 * tp + 0.5 * fp + fn) if tp + fp + fn else 0
    p = tp / (tp + fp) if tp + fp else 0
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, P=round(p, 3), R=round(tp / len(J), 3), F1w=round(f1, 4))
gx = shapely.total_bounds(GG[J])
for name in ['spor_arch', 'surf_arch']:
    d = json.load(open(f'/workspace/koz3_shared/bot-5/analysis/or_arch/{PFX}_{name}.json'))
    polys = []; confs = []
    for f in d['feats']:
        g = f['geom']; g = ast.literal_eval(g) if isinstance(g, str) else g
        polys.append(shape(g)); confs.append(float(f['confidence']))
    b = shapely.total_bounds(np.array(polys)) if polys else None
    # те же координаты, что у GT? если нет — перепроецировать 3857 → CRS GT
    if polys and not (b[0] < gx[2] + 5000 and b[2] > gx[0] - 5000):
        lon, lat = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True).transform((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        crs = f'EPSG:{32600 + int((lon + 180) // 6) + 1 if lat >= 0 else 32700 + int((lon + 180) // 6) + 1}'  # UTM по долготе, как у GT в table.pkl (проверка: границы ниже)
        tr = Transformer.from_crs('EPSG:3857', crs, always_xy=True).transform
        polys = [transform(tr, p) for p in polys]; print(name, 'reprojected 3857 ->', crs)
    print(name, 'n', len(polys), 'conf max', round(max(confs), 3) if confs else None, 'bounds', np.round(shapely.total_bounds(np.array(polys)), 1) if polys else None)
    out[name] = {'pre_declared_0.45': score(polys, confs, 0.45),
                 'curve_info_only': [score(polys, confs, t) for t in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6)]}
    print(name, out[name]['pre_declared_0.45']); [print('   ', c) for c in out[name]['curve_info_only']]
# стек A (боевой): строки mk=='or' класса arch в Герите, с их conf
cls, conf, reg, mk = T['cls'], T['conf'], T['reg'], T['mk']; k = T['CL'].index(G)
PP, JJ, SS = T['pairs'][(G, G)]; rows = np.flatnonzero((cls == k) & (reg == R) & (mk == 'or'))
pb = collections.defaultdict(list)
for a, j, s in zip(PP, JJ, SS):
    if reg[a] == R: pb[int(a)].append((int(j), float(s)))
def score_stack(thr):
    used = set(); tp = fp = 0
    for i in rows[np.argsort(-conf[rows])]:
        if conf[i] < thr: continue
        c = [(s, j) for j, s in pb.get(int(i), []) if j not in used]
        if c: used.add(max(c)[1]); tp += 1
        else: fp += 1
    fn = len(J) - tp
    return dict(thr=thr, tp=tp, fp=fp, fn=fn, F1w=round(2 * tp / (2 * tp + 0.5 * fp + fn), 4))
out['stack_A_or_rows'] = {'n': len(rows), 'conf_max': float(conf[rows].max()), 'curve': [score_stack(t) for t in (0.05, 0.1, 0.2, 0.45)]}
print('stack A', out['stack_A_or_rows'])
json.dump(out, open(f'/workspace/koz3_shared/bot-5/analysis/or_arch/score_or_{PFX}.json', 'w'), ensure_ascii=False, indent=1)
