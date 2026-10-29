#!/usr/bin/env python3
"""OOF D-FINE Or (koz3-dfine-or): P/R по классам при порогах, recall по площади, покрытие крупных городищ.
Матчинг = официальный F1Calculator._soft_match (koz3-1.5.0, bot-2/repo/materials/compute_metrics.py):
cov>=0.5 по меньшей фигуре + representative_point пред. внутри GT, один-к-одному, жадно по confidence; в UTM.
Сопоставление внутри (участок, класс). Только Or-разметка 6 Or-участков."""
import json, glob, sys, collections
sys.path.insert(0, '/workspace/koz3_shared/bot-2/repo/materials')
import compute_metrics as cm
from shapely.geometry import shape
OOF = sys.argv[1] if len(sys.argv) > 1 else '/workspace/kaggle_out/koz3-dfine-or/oof_dfine_or.geojson'
RU = {'архитектуры': 'arkhitektury', 'городища': 'gorodishcha', 'фортификации': 'fortifikatsii',
      'курганы_целые': 'kurgany_tselye', 'курганы_поврежденные': 'kurgany_povrezhdennye'}
REG = ['008_Герите', '014_Лешки', '033_Елизаветинское', '056_Фалхан', '093_Старый_Барах', '185_Изборск']
FOLD = {'008_Герите': 0, '014_Лешки': 0, '056_Фалхан': 0, '093_Старый_Барах': 0, '033_Елизаветинское': 1, '185_Изборск': 2}
calc = cm.F1Calculator()
utm = lambda g: cm._to_utm(g, 'webmercator')

def load_gt(markup='Or'):
    gt = []
    for r in REG:
        for f in glob.glob(f'/workspace/data/train-expds-3/{r}/06_*/*_{markup}_*.geojson'):
            cls = RU[f.split(f'_{markup}_')[-1][:-8]]
            for i, ft in enumerate(json.load(open(f))['features']):
                if not ft.get('geometry'): gt.append(dict(region=r, cls=cls, geom=None, area=None, i=i)); continue
                g = utm(shape(ft['geometry']).buffer(0))
                gt.append(dict(region=r, cls=cls, geom=g, area=g.area, i=i, file=f.split('/')[-1]))
    return gt

gt = load_gt()
pr = []
for ft in json.load(open(OOF))['features']:
    p = ft['properties']
    pr.append(dict(region=p['region_name'], cls=p['class_name'], conf=float(p['confidence']), fold=p['fold'],
                   geom=utm(shape(ft['geometry']).buffer(0))))
out = {'oof': OOF, 'n_pred': len(pr), 'n_gt_total': len(gt), 'n_gt_geom': sum(g['geom'] is not None for g in gt)}
out['gt_by_region_cls'] = dict(collections.Counter(f"{g['region']}|{g['cls']}" for g in gt))
out['pred_by_region_cls'] = dict(collections.Counter(f"{p['region']}|{p['cls']}" for p in pr))
conf = sorted(p['conf'] for p in pr); out['conf_min'], out['conf_max'] = conf[0], conf[-1]
out['conf_median'] = conf[len(conf) // 2]

def match(thr, regions, classes=None):
    """-> {cls: tp,fp,fn}, set matched gt ids"""
    res = collections.defaultdict(lambda: [0, 0, 0]); hit = set()
    for r in regions:
        for c in set([g['cls'] for g in gt if g['region'] == r] + [p['cls'] for p in pr if p['region'] == r]):
            G = [(g['geom'], 1.0) for g in gt if g['region'] == r and g['cls'] == c and g['geom'] is not None]
            Gi = [k for k, g in enumerate(gt) if g['region'] == r and g['cls'] == c and g['geom'] is not None]
            P = [(p['geom'], p['conf']) for p in pr if p['region'] == r and p['cls'] == c and p['conf'] >= thr]
            tp, fp, fn, m = calc._soft_match(P, G)
            for _, gg in m: hit.add(Gi[[id(G[j][0]) for j in range(len(G))].index(id(gg))])
            res[c][0] += tp; res[c][1] += fp; res[c][2] += fn
    return res, hit

def agnostic_hit(thr, regions):
    """класс-агностичное покрытие: GT накрыт хоть одним пред. любого класса (cov>=0.5 + внутр. точка), без 1-к-1"""
    h = set()
    for k, g in enumerate(gt):
        if g['geom'] is None or g['region'] not in regions: continue
        for p in pr:
            if p['region'] == g['region'] and p['conf'] >= thr and calc._match_polygon_gt(p['geom'], g['geom'])[0]:
                h.add(k); break
    return h

THR = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
bins = lambda a: '<500' if a < 500 else ('500-5000' if a <= 5000 else '>5000')
scopes = {'all6': REG, 'no_Gerite': [r for r in REG if r != '008_Герите']}
out['per_threshold'] = {}
for sn, regs in scopes.items():
    for t in THR:
        res, hit = match(t, regs)
        row = {}
        for c, (tp, fp, fn) in sorted(res.items()):
            row[c] = dict(tp=tp, fp=fp, fn=fn, P=round(tp / (tp + fp), 4) if tp + fp else None,
                          R=round(tp / (tp + fn), 4) if tp + fn else None,
                          F1_koz=round(2 * (tp / (tp + .5 * fp + 1e-9)) * (tp / (tp + fn + 1e-9)) / ((tp / (tp + .5 * fp + 1e-9)) + (tp / (tp + fn + 1e-9)) + 1e-9), 4))
        ab = collections.defaultdict(lambda: [0, 0]); ag = agnostic_hit(t, regs) if t in (0.05, 0.3) else None
        abg = collections.defaultdict(lambda: [0, 0])
        for k, g in enumerate(gt):
            if g['geom'] is None or g['region'] not in regs: continue
            b = bins(g['area']); ab[b][1] += 1; ab[b][0] += k in hit
            if ag is not None: abg[b][1] += 1; abg[b][0] += k in ag
        row['_recall_by_area_same_class'] = {b: f'{v[0]}/{v[1]}' for b, v in sorted(ab.items())}
        if ag is not None: row['_recall_by_area_any_class'] = {b: f'{v[0]}/{v[1]}' for b, v in sorted(abg.items())}
        out['per_threshold'][f'{sn}@{t}'] = row
# по фолдам (порог 0.05)
out['per_fold@0.05'] = {}
for f in (0, 1, 2):
    regs = [r for r in REG if FOLD[r] == f]
    res, _ = match(0.05, regs)
    out['per_fold@0.05'][f] = {c: dict(tp=v[0], fp=v[1], fn=v[2]) for c, v in res.items()}
# GT-объекты > 5000 м² (Or) — каждый
big = []
_, hit05 = match(0.05, REG); ag05 = agnostic_hit(0.05, REG)
for k, g in enumerate(gt):
    if g['geom'] is not None and g['area'] > 5000:
        best = max([(p['conf'], round(cm._polygon_coverage(p['geom'], g['geom']), 3), p['cls']) for p in pr
                    if p['region'] == g['region'] and p['geom'].intersects(g['geom'])], default=None)
        big.append(dict(region=g['region'], cls=g['cls'], file=g['file'], idx=g['i'], area=round(g['area'], 1),
                        matched_same_class_at_0_05=k in hit05, covered_any_class_at_0_05=k in ag05,
                        best_intersecting_pred_conf_cov_cls=best))
out['gt_gt5000_or'] = big
# Перекрытие Or-OOF с крупными (>5000) городищами других разметок на тех же участках (Ae/SpOr) — справочно
other = []
for mk in ('Ae', 'SpOr'):
    for g in load_gt(mk):
        if g['geom'] is None or g['cls'] != 'gorodishcha' or g['area'] <= 5000: continue
        ps = [p for p in pr if p['region'] == g['region'] and p['geom'].intersects(g['geom'])]
        other.append(dict(markup=mk, region=g['region'], idx=g['i'], area=round(g['area'], 1),
                          any_pred_hit_rule=any(calc._match_polygon_gt(p['geom'], g['geom'])[0] for p in ps),
                          gor_pred_hit_rule=any(p['cls'] == 'gorodishcha' and calc._match_polygon_gt(p['geom'], g['geom'])[0] for p in ps),
                          n_intersecting=len(ps), max_conf_intersecting=max([p['conf'] for p in ps], default=None)))
out['other_markup_gor_gt5000_same_regions'] = other
print(json.dumps(out, ensure_ascii=False, indent=1))
