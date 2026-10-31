"""Точная честная оценка микрокругов ПОСЛЕ порогов (как twins_exact.py): пороги групп = sweep_groups по кривым базы
на обучающих фолдах; на удержанных участках остаются conf >= порога (unseen=drop); затем фигура оставленного кандидата
заменяется кругом радиуса r в representative_point (conf не меняется, порог не меняется); официальное жадное сопоставление
(_match_quality) только по оставленным. merge_within_markup не моделируется (одинаково для всех вариантов)."""
import sys, os, json, time, resource, pickle, gc
resource.setrlimit(resource.RLIMIT_AS, (3_800_000_000, 3_800_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np, shapely
from shapely.strtree import STRtree
from collections import Counter
from common import *
from expds.metric import ClassCurve, sweep_groups, curve_where
from expds import metric as M
mod = M.load_official(METRIC); calc = mod.F1Calculator()
OUT = '/workspace/koz3_shared/bot-6/work/localize/'
T = pickle.load(open(W/'work/table.pkl', 'rb'))
CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}
cls, conf, reg, grp, cx, cy = T['cls'], T['conf'], T['reg'], T['grp'], T['cx'], T['cy']
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
PL0 = {}
for (pc, gc_), (P, J, S) in T['pairs'].items():
    d = {}; o = np.lexsort((J, P))
    for p, j, s in zip(P[o], J[o], S[o]): d.setdefault(int(p), []).append((int(j), float(s)))
    PL0[(pc, gc_)] = d
del T['pairs']; gc.collect()
GREG = {c: Counter(T['greg'][c]) for c in CL}
grp_s = grp.astype(str)
GTW = pickle.load(open(W/'work/gt_wkb.pkl', 'rb')); GT = {c: shapely.from_wkb(v) if len(v) else np.array([]) for c, v in GTW.items()}; del GTW
GTREE = {c: STRtree([x.buffer(calc.r_tolerance) if x.geom_type == 'Point' else x for x in GT[c]]) for c in CL if len(GT[c])}

def match(order_idx, c, PL):
    hit = np.zeros(len(order_idx), bool); matched = set()
    for r, p in enumerate(order_idx):
        lst = PL.get((CL[cls[p]], c), {}).get(int(p))
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in lst:
            if j in matched: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: matched.add(bj); hit[r] = True
    return hit
CV = {}
for c in CL:
    ii = np.flatnonzero(cls == cid[c]); o = ii[np.argsort(-conf[ii], kind='stable')]
    CV[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=match(o, c, PL0), pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                       gt_regions=GREG[c], pred_group=list(grp[o]))

gt_regions = set().union(*[set(GREG[c]) for c in CL])
regions = {r for i in range(len(CL)) for r in np.unique(reg[cls == i])} | gt_regions
boxes = json.loads((ROOT/'notes'/'las_boxes_59.json').read_text(encoding='utf-8'))
optical = {r for r in gt_regions if r} - set(boxes)
new = {l.strip() for l in (ROOT/'notes'/'regions_las_new.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
fams = sorted({site_family(r) for r in regions})
def rand_folds(seed, k=10):
    rng = np.random.default_rng(seed); f = list(fams); rng.shuffle(f)
    b = {x: i % k for i, x in enumerate(f)}; out = [set() for _ in range(k)]
    for r in regions: out[b[site_family(r)]].add(r)
    return [x for x in out if x]
FOLDSETS = [('k10', family_folds(regions, site_family, 10)), ('lofo', family_folds(regions, site_family, None))] + \
           [(f'rand{sd}', rand_folds(sd)) for sd in range(5)]
regarr = reg.astype(object)
KEPT = {}
t = time.time()
for name, folds in FOLDSETS:
    every = set().union(*folds); KEPT[name] = []
    for held in folds:
        res = sweep_groups({c: curve_where(v, every - held) for c, v in CV.items()})
        tau = {(g, cid[c]): gv['threshold'] for c, v in res.items() for g, gv in v['groups'].items()}
        hi = np.flatnonzero(np.isin(regarr, list(held)) & (cls >= 0))
        th = np.array([tau.get((grp_s[i], int(cls[i])), np.inf) for i in hi])
        KEPT[name].append((held, hi[conf[hi] >= th]))
allkept = np.unique(np.concatenate([k for v in KEPT.values() for _, k in v]))
print('thresholds', round(time.time() - t), 's; kept union', len(allkept), flush=True)
GW = pickle.load(open(W/'work/geom_wkb.pkl', 'rb'))
G = dict(zip(allkept.tolist(), shapely.from_wkb(GW[allkept]))); del GW; gc.collect()

def counts(PL, kept_list):
    cnt = {}
    for held, keep in kept_list:
        for c in CL:
            per = cnt.setdefault(c, {})
            for r in held:
                n = GREG[c].get(r, 0)
                if n: per.setdefault(r, [0, 0, 0])[2] += int(n)
            ii = keep[cls[keep] == cid[c]]; o = ii[np.argsort(-conf[ii], kind='stable')]
            for p, h in zip(o, match(o, c, PL)): per.setdefault(reg[p], [0, 0, 0])[0 if h else 1] += 1
    return cnt
def repair(newgeom):
    touched = {CL[cls[p]] for p in newgeom}
    PL = {k: (dict(v) if k[0] in touched else v) for k, v in PL0.items()}
    for p, g in newgeom.items():
        pc = CL[cls[p]]
        for gc_ in (KUR if pc in KUR else [pc]):
            PL.setdefault((pc, gc_), {}).pop(p, None)
            if gc_ not in GTREE: continue
            lst = []
            for j in GTREE[gc_].query(g):
                ok, s = calc._match_quality(g, GT[gc_][j])
                if ok: lst.append((int(j), float(s)))
            if lst: PL[(pc, gc_)][p] = sorted(lst)
    return PL
def anchor(g):
    c = g.centroid  # как post.micro_circles (пакет bot-7): центроид, если он внутри, иначе representative_point
    return c if g.contains(c) else g.representative_point()
def micro(classes, r=0.25):
    # радиус r — метры на земле; кэш в EPSG:3857 -> делим на масштаб cos(lat), как post._ground_scale
    ids = [cid[c] for c in classes]; out = {}
    for p in allkept:
        if cls[p] not in ids: continue
        g = G[int(p)]
        if g.geom_type not in ('Polygon', 'MultiPolygon') or g.is_empty: continue
        a = anchor(g); s = float(np.cos(np.arctan(np.sinh(a.y / 6378137.0))))
        out[int(p)] = a.buffer(r / s, quad_segs=8)
    return out
NOFP = [c for c in CL if c != 'finds_points']
VARS = {'A': lambda: {}, 'MC_all': lambda: micro(NOFP), 'MC_kur': lambda: micro(KUR), 'MC05_kur': lambda: micro(KUR, 0.5),
        'MC_fort': lambda: micro(['fortifikatsii']), 'MC_gor': lambda: micro(['gorodishcha']), 'MC_arch': lambda: micro(['arkhitektury']),
        'MC_op': lambda: micro(['object_poly']), 'MC_nokur': lambda: micro([c for c in NOFP if c not in KUR]),
        'MC05_gor': lambda: micro(['gorodishcha'], 0.5), 'MC05_fort': lambda: micro(['fortifikatsii'], 0.5),
        'MC05_arch': lambda: micro(['arkhitektury'], 0.5),
        'MC_kurfort': lambda: {**micro(KUR), **micro(['fortifikatsii'])},
        'MC_kurfortarch': lambda: {**micro(KUR), **micro(['fortifikatsii']), **micro(['arkhitektury'])}}
only = [a for a in sys.argv[1:]]
R = {}; C10 = {}
W8 = {'kurgany_povrezhdennye': 30, 'kurgany_tselye': 20, 'gorodishcha': 15}
for v, fn in VARS.items():
    if only and v not in only and v != 'A': continue
    t = time.time(); ng = fn(); PL = repair(ng) if ng else PL0
    R[v] = {'moved': len(ng)}
    for name, _ in FOLDSETS:
        cnt = counts(PL, KEPT[name])
        if name == 'k10':
            C10[v] = cnt; s, per = score_counts(cnt)
            R[v].update(all=s, new34=score_counts(cnt, new)[0], optical=score_counts(cnt, optical)[0], per={c: per[c] for c in CL})
        else: R[v][name] = score_counts(cnt)[0]
    del ng, PL; gc.collect()
    if v != 'A':
        A, B = R['A'], R[v]
        bs = paired_bootstrap_counts(C10[v], C10['A'], site_family, 1000)
        d = B['all'] - A['all']
        p4 = {c: w * (B['per'][c]['f1'] - A['per'][c]['f1']) / 100 for c, w in W8.items()}
        drand = [B[f'rand{s}'] - A[f'rand{s}'] for s in range(5)]
        rule = {'P1': d >= 0.002, 'P2': bs['p_gt0'] >= 0.85, 'P3': (B['lofo'] - A['lofo'] > 0) and sum(x > 0 for x in drand) >= 4,
                'P4': all(x >= -0.0005 for x in p4.values()) and B['new34'] >= A['new34'] - 0.001 and B['optical'] >= A['optical'] - 0.001,
                'P5_package_only': d < 0.006}
        chg = {c: (round(B['per'][c]['tp'] - A['per'][c]['tp']), round(B['per'][c]['fp'] - A['per'][c]['fp'])) for c in CL if B['per'][c]['tp'] != A['per'][c]['tp'] or B['per'][c]['fp'] != A['per'][c]['fp']}
        B['vsA'] = dict(delta=d, d_new34=B['new34'] - A['new34'], d_optical=B['optical'] - A['optical'], d_lofo=B['lofo'] - A['lofo'],
                        d_rand=drand, boot=bs, P_A_gt_B=1 - bs['p_gt0'], P4=p4, rule=rule, dTP_dFP=chg)
        print(v, 'moved', B['moved'], 'delta %.6f new34 %+.6f opt %+.6f lofo %+.6f rand %s P(A>B) %.3f CI90 [%+.5f;%+.5f] dTP/dFP %s rule %s' % (
            d, B['vsA']['d_new34'], B['vsA']['d_optical'], B['vsA']['d_lofo'], [round(x, 5) for x in drand], 1 - bs['p_gt0'], bs['p5'], bs['p95'], chg, rule),
            round(time.time() - t), 's', flush=True)
    else:
        print('A', round(R['A']['all'], 6), 'new34', round(R['A']['new34'], 6), 'opt', round(R['A']['optical'], 6), 'lofo', round(R['A']['lofo'], 6), round(time.time() - t), 's', flush=True)
    json.dump(R, open(OUT + os.environ.get('LOC_OUT', 'localize_exact') + '.json', 'w'), default=float, indent=1, ensure_ascii=False)
print('done')
