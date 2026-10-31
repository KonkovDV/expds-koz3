"""Честная оценка замены кандидатов finds_points / object_poly(maggr) на выбранных участках.
Остальные классы — из /workspace/rerank/work/base_honest.pkl (тот же стек, те же фолды).
Код метрики: bot-10/repo (unseen keep|drop, EXPDS_MATCH_SCOPE=site), официальный compute_metrics.py."""
import sys, os, json, pickle
import numpy as np, shapely
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo')
sys.path.insert(0, '/workspace/koz3_shared/bot-10/repo/scripts')
from expds import metric as M
from expds.naming import site_family
METRIC = '/workspace/expds-koz3/materials/compute_metrics.py'
mod = M.load_official(METRIC); calc = mod.F1Calculator()
M._SHM_CACHE_DIR = __import__('pathlib').Path('/nonexistent_ro/x')
GEO = ['finds_points', 'object_poly']
S = pickle.load(open('/workspace/koz3_shared/bot-3/geo_gap/sub_geo.pkl', 'rb'))
BASE = pickle.load(open('/workspace/rerank/work/base_honest.pkl', 'rb'))
FOLDS = BASE['folds']
REGIONS = set().union(*FOLDS)
NEW = {l.strip() for l in open('/workspace/expds-koz3/notes/regions_las_new.txt', encoding='utf-8') if l.strip()}
_boxes = json.load(open('/workspace/expds-koz3/notes/las_boxes_59.json', encoding='utf-8'))
OPTICAL = REGIONS - set(_boxes)

GTS = {c: [(g, 1.0, r, '') for g, r in zip(shapely.from_wkb(S['gt_wkb'][c]), S['greg'][c])] for c in GEO}

def stack_preds(c):
    s = S[c]; g = shapely.from_wkb(s['wkb'])
    return [(g[i], float(s['conf'][i]), str(s['reg'][i]), str(s['grp'][i]), str(s['mk'][i])) for i in range(len(g))]

def file_preds(path):
    fc = json.load(open(path))
    out = M._collect(mod, calc, fc, False)
    mk = {}
    for c, v in out.items():
        mk[c] = [(g, conf, reg, grp, 'maggr') for g, conf, reg, grp in v]
    return mk

def swap(c, regs, new_preds, keep_group=lambda grp: grp != 'maggr'):
    """стек класса c, где на участках regs maggr-кандидаты заменены new_preds."""
    base = [p for p in stack_preds(c) if not (p[2] in regs and not keep_group(p[3]))]
    return base + [p for p in new_preds if p[2] in regs]

def curves_of(preds_by_cls, scope='global'):
    os.environ['EXPDS_MATCH_SCOPE'] = scope
    cv = M.match_collected(calc, {c: [p[:4] for p in v] for c, v in preds_by_cls.items()},
                           {c: GTS[c] for c in preds_by_cls})
    return {c: cv[c] for c in preds_by_cls}

def counts_with(curves, folds=FOLDS, unseen='drop'):
    cs, info = M.honest_counts(curves, folds, unseen=unseen)
    full = {c: v for c, v in BASE['counts'].items()}
    full.update(cs)
    return full, info

def scores(counts):
    return [M.score_counts(counts)[0], M.score_counts(counts, NEW)[0], M.score_counts(counts, OPTICAL)[0]]

from honest_bootstrap import random_family_folds
LOFO = M.family_folds(REGIONS, site_family, None)
RAND = [random_family_folds(REGIONS, 10, s) for s in range(5)]

def criteria(cA, cB, unseen='drop'):
    """П1–П5 для A (вариант) против B (база): кривые обоих — dict классов GEO."""
    a, ia = counts_with(cA, unseen=unseen); b, ib = counts_with(cB, unseen=unseen)
    sa, sb = scores(a), scores(b)
    d = sa[0] - sb[0]
    pb = M.paired_bootstrap_counts(a, b, site_family, n=1000)
    lofo = scores(counts_with(cA, LOFO, unseen)[0])[0] - scores(counts_with(cB, LOFO, unseen)[0])[0]
    rnd = [scores(counts_with(cA, f, unseen)[0])[0] - scores(counts_with(cB, f, unseen)[0])[0] for f in RAND]
    pa, pbb = M.score_counts(a)[1], M.score_counts(b)[1]
    den = sum(M.CLASS_WEIGHTS[c] for c in pa if pa[c]['n_gt'] > 0)
    wd = {c: M.CLASS_WEIGHTS[c] * (pa[c]['f1'] - pbb[c]['f1']) / den for c in pa}
    res = dict(base=sb, var=sa, d_honest10=d, boot=pb, d_lofo=lofo, d_rand=rnd,
               w_delta=wd, per_class_var={c: pa[c] for c in GEO}, per_class_base={c: pbb[c] for c in GEO},
               unseen_info=(ia, ib))
    res['P1'] = d >= 0.002
    res['P2'] = pb.get('p_gt0', 0) >= 0.85
    res['P3'] = lofo > 0 and sum(x > 0 for x in rnd) >= 4
    res['P4'] = all(wd[c] >= -0.0005 for c in ['kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha']) \
        and sa[1] >= sb[1] - 0.001 and sa[2] >= sb[2] - 0.001
    res['P5_below_MDE'] = d < 0.006
    return res
