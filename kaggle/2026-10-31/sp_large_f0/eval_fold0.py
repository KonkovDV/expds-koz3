"""sp_large fold0 stop metrics (PLAN_SP_LARGE). Official compute_metrics matching (expds.metric.build_curves), lidar untouched.
usage: python eval_fold0.py new.geojson [control.geojson]"""
import json, sys, numpy as np
sys.path.insert(0, '/workspace/koz3/wt/sp-large-f0')
from expds import metric as M
from expds.crs import utm_for_lonlat
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree
from pyproj import Transformer
D = '/workspace/koz3/out/sp_large-f0/'
MET = '/workspace/koz3/wt/sp-large-f0/materials/compute_metrics.py'
R = set(json.load(open(D + 'fold0_split.json'))['fold0_regions'])
K = ('kurgany_povrezhdennye', 'kurgany_tselye')
SUB_THR = {'kurgany_povrezhdennye': 0.3645, 'kurgany_tselye': 0.3725}
g = json.load(open('/workspace/koz3/w5data/gt_train_v2.geojson'))
gt = {'type': 'FeatureCollection', 'features': [f for f in g['features'] if f['properties'].get('region_name') in R
      and f['properties'].get('markup_type') == 'sp_or' and f['properties'].get('class_name') in K]}
to4326 = Transformer.from_crs(3857, 4326, always_xy=True)
def area_m2(geom):
    c = geom.representative_point(); lon, lat = to4326.transform(c.x, c.y)
    t = Transformer.from_crs(3857, utm_for_lonlat(lon, lat), always_xy=True)
    return transform(t.transform, geom).area
dmg = [(shape(f['geometry']),) for f in gt['features'] if f['properties']['class_name'] == K[0]]
dmg = [(g, area_m2(g)) for (g,) in dmg]
big = [g for g, a in dmg if 5000 <= a <= 10000]
big1k = [g for g, a in dmg if a >= 1000]
def prep(fc):
    fs = []
    for f in fc['features']:
        p = dict(f['properties']); p.pop('threshold_group', None); p['markup_type'] = 'sp_or'
        if p.get('region_name') in R and p.get('class_name') in K: fs.append({**f, 'properties': p})
    return {'type': 'FeatureCollection', 'features': fs}
def hits(fc, thr, big=big):
    pts = [shape(f['geometry']).centroid for f in fc['features'] if f['properties']['class_name'] == K[0]
           and f['properties']['confidence'] >= thr]
    if not pts: return 0
    tr = STRtree(pts)
    return sum(1 for b in big if any(b.contains(pts[i]) for i in tr.query(b)))
def evaluate(name, fc, thr_override=None):
    fc = prep(fc); cur = M.build_curves(fc, gt, MET); out = {'n_circles': len(fc['features'])}
    for c in K:
        cv = cur[c]; taus, f1 = M.curve_table(cv); i = int(np.argmax(f1))
        sub = SUB_THR[c]; kept = cv.conf >= sub; tp = int(cv.hit[kept].sum())
        out[c] = {'n_gt': cv.n_gt, 'best_thr': float(taus[i]), 'f1_best': float(f1[i]), 'kept_best': int((cv.conf >= taus[i]).sum()),
                  'f1_at_sub_thr': float(M._f1(tp, int(kept.sum()) - tp, cv.n_gt - tp)), 'kept_sub': int(kept.sum())}
    td = out[K[0]]['best_thr']
    out['hits_5_10k_damaged'] = {'n_gt': len(big), 'no_thr': hits(fc, 0.0), 'at_sub_thr': hits(fc, SUB_THR[K[0]]),
                                 'at_own_best_thr': hits(fc, td if np.isfinite(td) else 9)}
    out['hits_ge1k_damaged_suppl'] = {'n_gt': len(big1k), 'no_thr': hits(fc, 0.0, big1k), 'at_sub_thr': hits(fc, SUB_THR[K[0]], big1k),
                                 'at_own_best_thr': hits(fc, td if np.isfinite(td) else 9, big1k)}
    out['circles_past_thr'] = sum(out[c]['kept_best'] for c in K)
    return out
res = {'baseline_sp_focal_oof': evaluate('base', json.load(open(D + 'baseline_spfocal_fold0.geojson')))}
for p in sys.argv[1:]: res[p.rsplit('/', 1)[-1]] = evaluate(p, json.load(open(p)))
json.dump(res, open(D + 'fold0_metrics.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False, indent=1))
