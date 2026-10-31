"""Общее для geo_snap (bot-3): кандидаты S (pix LOFO + стек, правило eval_pix.variant_S), честные маски honest-10,
точное жадное сопоставление (официальный _match_quality, участок к участку)."""
import os, sys, json, collections
GG = '/workspace/koz3_shared/bot-3/geo_gap'; W = '/workspace/koz3_shared/bot-3/geo_snap'
sys.path.insert(0, GG); _cwd = os.getcwd(); os.chdir(GG)
from evlib import *
os.chdir(_cwd)
from shapely.strtree import STRtree
CL = ['finds_points', 'object_poly']
PIXF = '/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson'
PIX = {c: [(g, cf, r, grp, 'pix') for g, cf, r, grp, _ in v] for c, v in file_preds(PIXF).items()}
STACK = {c: stack_preds(c) for c in CL}
mt_of = lambda grp: grp.replace('_pix', '')

def variant_S(c, base, pix=None):
    pix = PIX[c] if pix is None else pix
    cov = {(p[2], mt_of(p[3])) for p in pix}
    return [p for p in base if (p[2], p[3]) not in cov] + list(pix)

SV = {c: variant_S(c, STACK[c]) for c in CL}

def kept_mask(P, c, unseen, scope):
    """порог группы для семьи held — оптимум sweep на остальных 9 фолдах (как honest_counts)."""
    cv = curves_of({c: P}, scope)[c]
    keep = np.zeros(len(P), bool); every = set().union(*FOLDS)
    for held in FOLDS:
        res = M.sweep_groups({c: M.curve_where(cv, every - held)})
        taus = {g: gv['threshold'] for g, gv in res[c]['groups'].items()}
        for i in range(len(P)):
            if P[i][2] in held:
                t = taus.get(P[i][3])
                keep[i] = (P[i][1] >= t) if t is not None else (unseen == 'keep')
    return keep

def greedy(P, c, idx=None):
    """официальное жадное сопоставление по убыванию conf, эталон своего участка; idx — подмножество кандидатов."""
    G = GTS[c]; gg = [x[0] for x in G]; tree = STRtree(gg)
    idx = range(len(P)) if idx is None else idx
    order = sorted(idx, key=lambda i: P[i][1], reverse=True)
    matched, used = {}, set()
    for i in order:
        pg = P[i][0]
        if c == 'object_poly' and pg.geom_type != 'Polygon': continue
        q = pg if c == 'object_poly' else pg.centroid.buffer(calc.r_tolerance)
        best, bj = -np.inf, -1
        for j in sorted(int(k) for k in tree.query(q)):
            if j in used or G[j][2] != P[i][2]: continue
            ok, s = calc._match_quality(pg, gg[j])
            if ok and s > best + 1e-9: best, bj = s, j
        if bj >= 0: used.add(bj); matched[i] = bj
    return matched
