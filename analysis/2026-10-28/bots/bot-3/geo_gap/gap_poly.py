"""Разбор FN object_poly (стек, честно: пороги фолда без своей семьи; unseen drop и keep)."""
import json, collections
from evlib import *
from shapely.strtree import STRtree
c = 'object_poly'
P = stack_preds(c); G = GTS[c]
gg = [x[0] for x in G]; greg = [x[2] for x in G]
order = sorted(range(len(P)), key=lambda i: P[i][1], reverse=True)
tree = STRtree(gg)
# жадное официальное сопоставление с запоминанием эталона
matched = {}; used = set()
for i in order:
    pg = P[i][0]
    if pg.geom_type != 'Polygon': continue
    best, bj = -np.inf, -1
    for j in sorted(int(k) for k in tree.query(pg)):
        if j in used: continue
        ok, s = calc._match_quality(pg, gg[j])
        if ok and s > best + 1e-9: best, bj = s, j
    if bj >= 0: used.add(bj); matched[i] = bj
# пороги честно: для каждого фолда
cv = curves_of({c: P})[c]
def kept_mask(unseen):
    keep = np.zeros(len(P), bool)
    every = set().union(*FOLDS)
    for held in FOLDS:
        res = M.sweep_groups({c: M.curve_where(cv, every - held)})
        taus = {g: gv['threshold'] for g, gv in res[c]['groups'].items()}
        for i in range(len(P)):
            if P[i][2] in held:
                t = taus.get(P[i][3])
                keep[i] = (P[i][1] >= t) if t is not None else (unseen == 'keep')
    return keep
out = {}
cent = [p[0].centroid for p in P]
ptree = STRtree([p[0] for p in P])
# пары, проходящие геометрию (любой кандидат, любой порог)
okpair = collections.defaultdict(list)
for i, p in enumerate(P):
    for j in tree.query(p[0]):
        ok, s = calc._match_quality(p[0], gg[j])
        if ok: okpair[int(j)].append(i)
def q(v, ps=(10, 25, 50, 75, 90)):
    v = np.asarray(v, float); return {f'p{x}': round(float(np.percentile(v, x)), 3) for x in ps} | {'n': int(v.size)} if v.size else {'n': 0}
for unseen in ['drop', 'keep']:
    keep = kept_mask(unseen)
    tp_g = {j for i, j in matched.items() if keep[i]}
    fn = [j for j in range(len(G)) if j not in tp_g]
    cause = collections.Counter(); dist_near = []; ratio = []; covs = []; rp_in = []; fails = collections.Counter()
    by_mk = collections.Counter()
    per_reg = collections.defaultdict(collections.Counter)
    for j in fn:
        g = gg[j]
        near = [int(i) for i in ptree.query(g.buffer(5.0))]
        if not near:
            k = 'A_нет_кандидата_в_5м'
        elif okpair.get(j):
            ii = okpair[j]
            if any(keep[i] for i in ii):
                k = 'B_совпадает_но_отдан_другому_(жадность)'
            elif all(i in matched and matched[i] != j for i in ii):
                k = 'B2_совпадающий_занят_другим_эталоном'
            else:
                k = 'C_совпадает_но_ниже_порога' + ('_или_unseen' if unseen == 'drop' else '')
        else:
            k = 'D_рядом_но_геометрия_не_проходит'
            # ближайший кандидат по расстоянию
            ds = [g.distance(P[i][0]) for i in near]; i = near[int(np.argmin(ds))]
            pg = P[i][0]
            inter = pg.intersection(g).area; mn = min(pg.area, g.area)
            cov = inter / mn if mn > 0 else 0
            rp = g.buffer(1e-10).contains(pg.representative_point())
            dist_near.append(g.centroid.distance(pg.centroid)); ratio.append(pg.area / max(g.area, 1e-9)); covs.append(cov)
            fails['cov<0.5' if cov < 0.5 else 'cov_ok'] += 0
            if cov >= 0.5 and not rp: fails['cov_ok_но_точка_вне'] += 1
            elif cov < 0.5 and rp: fails['точка_внутри_но_cov<0.5'] += 1
            elif cov < 0.5 and not rp: fails['оба_не_проходят'] += 1
            else: fails['оба_ок?(ближайший_не_тот)'] += 1
            # лучший по cov среди всех в 5 м
        cause[k] += 1; per_reg[greg[j][:3]][k[0]] += 1
    out[unseen] = dict(n_gt=len(G), tp=len(tp_g), fn=len(fn), causes=dict(cause), D_fail=dict(fails),
                       D_centroid_dist_m=q(dist_near), D_area_ratio_pred_over_gt=q(ratio), D_cov=q(covs),
                       gt_area_fn=q([gg[j].area for j in fn]), gt_area_tp=q([gg[j].area for j in tp_g]),
                       cand_area_all=q([p[0].area for p in P]),
                       per_region={r: dict(v) for r, v in sorted(per_reg.items())})
    print(unseen, json.dumps(out[unseen], ensure_ascii=False), flush=True)
json.dump(out, open('gap_poly.json', 'w'), ensure_ascii=False, indent=1)
