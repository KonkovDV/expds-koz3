"""Городища без лидара: почему кандидат sp_or на эталоне не засчитан (официальная _match_quality)."""
import sys, json, pickle, collections
import numpy as np, shapely
from pathlib import Path
ROOT = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(ROOT))
from expds import metric as M
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; c = 'gorodishcha'; k = CL.index(c)
rows = np.flatnonzero(T['cls'] == k)
W = pickle.load(open('/workspace/rerank/work/geom_wkb.pkl', 'rb'))
geo = {int(i): shapely.from_wkb(W[i]) for i in rows}; del W
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
gt = json.load(open('/workspace/rerank/gt_train_v2.geojson')); gmk = []
for f in gt['features']:
    p = f.get('properties') or {}
    if mod.normalize_class_name(p.get('class_name', '')) == c and f.get('geometry'): gmk.append(p.get('markup_type'))
G = shapely.from_wkb(pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))[c]); greg = np.array(T['greg'][c])
boxes = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))
TH = 0.994085
out = collections.Counter(); det = []
for j in range(len(G)):
    if greg[j] in boxes: continue
    cand = [i for i in rows if T['reg'][i] == greg[j] and T['grp'][i] == 'sp_or' and geo[int(i)].intersects(G[j])]
    if not cand: out[f'{gmk[j]}|no_sp_or_mask_intersects'] += 1; continue
    best = None
    for i in cand:
        g = geo[int(i)]; inter = g.intersection(G[j]).area; cov = inter / min(g.area, G[j].area)
        rp = G[j].contains(g.representative_point()); ok = calc._match_quality(g, G[j])[0] if hasattr(calc, '_match_quality') else None
        r = dict(conf=float(T['conf'][i]), cov=round(cov, 3), rep_in=bool(rp), pred_area=round(g.area), gt_area=round(G[j].area),
                 pred_over_gt=round(g.area / G[j].area, 3), ok=bool(ok))
        if best is None or (r['ok'], r['conf']) > (best['ok'], best['conf']): best = r
    key = 'match_ok' if best['ok'] else ('cov_lt_0.5' if best['cov'] < 0.5 else ('rep_point_outside' if not best['rep_in'] else 'other'))
    key += '|above_thr' if best['conf'] >= TH else '|below_thr'
    out[f'{gmk[j]}|{key}'] += 1; det.append(dict(gt=j, mk=gmk[j], region=greg[j], **best))
res = {'counts': dict(sorted(out.items())), 'detail': det}
json.dump(res, open('/workspace/koz3_shared/bot-5/analysis/nolidar/gor_fail.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res['counts'], indent=1, ensure_ascii=False))
for d in det: print(d)
