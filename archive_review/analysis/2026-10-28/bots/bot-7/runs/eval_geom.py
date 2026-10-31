"""Честная проверка генератора геометрии (expds/geom_opt.py) на стеке вне обучения.

Протокол (точный, не по кривым): 10 фолдов семей (family_folds(site_family, 10), как
/workspace/rerank/base.py). Для каждого фолда пороги (группа, класс) — sweep_groups на
остальных фолдах по кривым ПРЕОБРАЗОВАННОГО до порога стека; затем пороги применяются к
участкам фолда, затем пост-шаг (склейка/разделение) и официальное сопоставление
(build_curves = _match_quality из compute_metrics.py) только на этих участках.
Итог — счётчики TP/FP/GT по участкам → score_counts.
Запуск: OMP_NUM_THREADS=2 nice -n 10 ../venv/bin/python eval_geom.py EXP1 EXP2 ...
"""
from common7 import *
import expds.metric as _m
from expds import geom_opt, post
from expds.metric import curve_where
import numpy as np, shapely, resource, os
from shapely.geometry import shape, mapping

UNSEEN = os.environ.get('UNSEEN', 'drop')
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
def kur(rule): return {c: dict(rule) for c in KUR}

EXPS = {
 'base': ({}, {}),
 # курганы: радиус диска в том же центре (сейчас 2 м, sp_focal 4 м)
 'kur_r0.1': ({'reshape': kur({'r_m': 0.1, 'anchor': 'centroid'})}, {}),
 'kur_r0.5': ({'reshape': kur({'r_m': 0.5, 'anchor': 'centroid'})}, {}),
 'kur_r1':   ({'reshape': kur({'r_m': 1.0, 'anchor': 'centroid'})}, {}),
 'kur_r2':   ({'reshape': kur({'r_m': 2.0, 'anchor': 'centroid'})}, {}),
 'kur_r3':   ({'reshape': kur({'r_m': 3.0, 'anchor': 'centroid'})}, {}),
 'kur_r5':   ({'reshape': kur({'r_m': 5.0, 'anchor': 'centroid'})}, {}),
 'kur_r10':  ({'reshape': kur({'r_m': 10.0, 'anchor': 'centroid'})}, {}),
 # курганы: склейка дублей после порога (между группами) и классическое объединение
 'kur_dd1':  ({}, {'dedup': kur({'radius_m': 1.0})}),
 'kur_dd2':  ({}, {'dedup': kur({'radius_m': 2.0})}),
 'kur_dd3':  ({}, {'dedup': kur({'radius_m': 3.0})}),
 'kur_dd5':  ({}, {'dedup': kur({'radius_m': 5.0})}),
 'kur_ddm1':  ({}, {'dedup': kur({'radius_m': 1.0, 'within': 'markup'})}),
 'kur_ddm2':  ({}, {'dedup': kur({'radius_m': 2.0, 'within': 'markup'})}),
 'kur_ddm3':  ({}, {'dedup': kur({'radius_m': 3.0, 'within': 'markup'})}),
 'kur_ddm5':  ({}, {'dedup': kur({'radius_m': 5.0, 'within': 'markup'})}),
 'kur_ddg2':  ({}, {'dedup': kur({'radius_m': 2.0, 'within': 'group'})}),
 'kur_r0.5_ddm2': ({'reshape': kur({'r_m': 0.5, 'anchor': 'centroid'})}, {'dedup': kur({'radius_m': 2.0, 'within': 'markup'})}),
 'gor_ddm_in': ({}, {'dedup': {'gorodishcha': {'radius_m': 10.0, 'inside': True, 'within': 'markup'}}}),
 'gor_rp0.5_ddm_in': ({'reshape': {'gorodishcha': {'r_m': 0.5, 'anchor': 'rp'}}}, {'dedup': {'gorodishcha': {'radius_m': 10.0, 'within': 'markup'}}}),
 'obj_ddm0.5': ({}, {'dedup': {'object_poly': {'radius_m': 0.5, 'within': 'markup'}}}),
 'obj_split3_post_ddm0.5': ({}, {'split': {'object_poly': {'min_area_m2': 12, 'max_k': 3, 'min_lobe_m2': 3, 'min_sep_m': 0.8, 'decay': 0.6, 'r_m': 0.05}}, 'dedup': {'object_poly': {'radius_m': 0.5, 'within': 'markup'}}}),
 'fort_ddm3': ({}, {'dedup': {'fortifikatsii': {'radius_m': 3.0, 'within': 'markup'}}}),
 'kur_union': ({}, {'merge_union': kur({})}),
 # итоговая рекомендация (зафиксирована по аналитике МК + таблице; см. REPORT.md, оговорка о выборе)
 'combo': ({'reshape': {**kur({'r_m': 0.5, 'anchor': 'centroid'}),
                        'gorodishcha': {'r_m': 0.5, 'anchor': 'polylabel', 'r_frac': 0.5}}},
           {'split': {'object_poly': {'min_area_m2': 12, 'max_k': 3, 'min_lobe_m2': 3, 'min_sep_m': 0.8, 'decay': 0.6, 'r_m': 0.05}}}),
 'kur_twin':  ({'twins': kur({})}, {}),
 'kur_twin_r0.5': ({'reshape': kur({'r_m': 0.5, 'anchor': 'centroid'}), 'twins': kur({})}, {}),
 'gor_twin':  ({'twins': {'gorodishcha': {}}}, {}),
 'obj_twin':  ({'twins': {'object_poly': {}}}, {}),
 'fort_twin': ({'twins': {'fortifikatsii': {}}}, {}),
 # городища
 'gor_rp0.5':  ({'reshape': {'gorodishcha': {'r_m': 0.5, 'anchor': 'rp'}}}, {}),
 'gor_pl0.5':  ({'reshape': {'gorodishcha': {'r_m': 0.5, 'anchor': 'polylabel'}}}, {}),
 'gor_core':   ({'reshape': {'gorodishcha': {'r_m': 0.5, 'anchor': 'polylabel', 'r_frac': 0.5}}}, {}),
 'gor_split3': ({'split': {'gorodishcha': {'min_area_m2': 2000, 'max_k': 3, 'min_lobe_m2': 300, 'min_sep_m': 20, 'decay': 0.6, 'r_m': 0.5}}}, {}),
 'gor_dd_in':  ({}, {'dedup': {'gorodishcha': {'radius_m': 10.0, 'inside': True}}}),
 'gor_union':  ({}, {'merge_union': {'gorodishcha': {}}}),
 # object_poly
 'obj_rp':     ({'reshape': {'object_poly': {'r_m': 0.05, 'anchor': 'rp'}}}, {}),
 'obj_pl':     ({'reshape': {'object_poly': {'r_m': 0.05, 'anchor': 'polylabel'}}}, {}),
 'obj_multipole_post': ({}, {'multipole': True}),
 'obj_split3': ({'split': {'object_poly': {'min_area_m2': 12, 'max_k': 3, 'min_lobe_m2': 3, 'min_sep_m': 0.8, 'decay': 0.6, 'r_m': 0.05}}}, {}),
 'obj_split3_post': ({}, {'split': {'object_poly': {'min_area_m2': 12, 'max_k': 3, 'min_lobe_m2': 3, 'min_sep_m': 0.8, 'decay': 0.6, 'r_m': 0.05}}}),
 'obj_dd0.5':  ({}, {'dedup': {'object_poly': {'radius_m': 0.5}}}),
 'obj_union':  ({}, {'merge_union': {'object_poly': {}}}),
}

def to_internal(fc):
    out = []
    for f in fc['features']:
        d = dict(f['properties']); d['geom'] = shape(f['geometry']); out.append(d)
    return out

def to_fc(feats):
    fs = []
    for d in feats:
        p = {k: v for k, v in d.items() if k != 'geom'}
        fs.append({'type': 'Feature', 'properties': p, 'geometry': mapping(d['geom'])})
    return {'type': 'FeatureCollection', 'features': fs}

def run_post(feats, cfg):
    if cfg.get('multipole'):
        feats = post.multipole_object_poly(feats)
    rest = {k: v for k, v in cfg.items() if k != 'multipole'}
    return geom_opt.optimize(feats, rest) if rest else feats

def evaluate(name, base_int, gt, folds):
    pre, pst = EXPS[name]
    feats = geom_opt.optimize(base_int, pre) if pre else base_int
    fc = to_fc(feats)
    curves = build_curves(fc, gt, METRIC)
    ins = insample(curves)[0]
    approx = score_counts(honest_counts(curves, folds, unseen=UNSEEN)[0])[0]
    every = set().union(*folds)
    counts = {c: {} for c in curves}
    norm = _m.load_official(METRIC).normalize_class_name
    for held in folds:
        res = sweep_groups({c: curve_where(cv, every - held) for c, cv in curves.items()})
        taus = {}
        for c, v in res.items():
            for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
        kept = []
        for f, d in zip(fc['features'], feats):
            p = f['properties']
            if p.get('region_name') not in held: continue
            t = taus.get(_m.threshold_group(p), {}).get(norm(p.get('class_name')))
            if t is None and UNSEEN == 'drop': continue   # как honest_counts(unseen="drop") — по умолчанию
            if t is not None and float(p.get('confidence', 1.0)) < t: continue
            kept.append((f, d))
        if pst:
            kfc = to_fc(run_post([d for _, d in kept], pst))
        else:
            kfc = {'type': 'FeatureCollection', 'features': [f for f, _ in kept]}
        ghe = {'type': 'FeatureCollection', 'features': [g for g in gt['features'] if (g.get('properties') or {}).get('region_name') in held]}
        hc = build_curves(kfc, ghe, METRIC)
        for c, cv in hc.items():
            per = counts.setdefault(c, {})
            for r, n in cv.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            for r, h in zip(cv.pred_region, cv.hit): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return ins, approx, counts

if __name__ == '__main__':
    names = sys.argv[1:] or list(EXPS)
    st = load(W / 'stack_sub6000.geojson'); gt = load('/workspace/rerank/gt_train_v2.geojson')
    new, opt = sets(gt)
    base_int = to_internal(st); del st
    regions = {f['region_name'] for f in base_int} | {(g.get('properties') or {}).get('region_name') for g in gt['features']}
    folds = family_folds(regions, site_family, 10)
    store = W / ('eval_geom.pkl' if UNSEEN == 'drop' else f'eval_geom_{UNSEEN}.pkl')
    db = pickle.load(open(store, 'rb')) if store.exists() else {}
    import gc
    for n in names:
        t = time.time(); _m._UTM_GEOM_CACHE.clear(); gc.collect()
        ins, approx, counts = evaluate(n, base_int, gt, folds)
        s = [score_counts(counts)[0], score_counts(counts, new)[0], score_counts(counts, opt)[0]]
        per = score_counts(counts)[1]
        db[n] = {'ins': ins, 'approx_honest': approx, 'exact_honest': s, 'counts': counts}
        pickle.dump(db, open(store, 'wb'))
        print(f"{n:20s} ins {ins:.5f} honest~ {approx:.5f} | exact all {s[0]:.5f} new {s[1]:.5f} opt {s[2]:.5f} | "
              + ' '.join(f"{c[:6]} {v['f1']:.4f}" for c, v in per.items()) + f" | {time.time()-t:.0f}s rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f}MB", flush=True)
