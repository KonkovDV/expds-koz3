"""Уточнение локализации без растров на кэше /workspace/rerank (база A = stack_arch).
Политика: меняется геометрия только кандидатов с conf >= min по фолдам порога своей (группы, класса)
(на инференсе порог известен); остальные кандидаты и их пары — как в базе. Пары пересчитаны официальным _match_quality.
Варианты:
  Cr_<cls>: центр кандидата -> conf-взвешенное среднее центроидов кандидатов того же класса (курганы — оба класса вместе)
            того же участка в радиусе r м (сдвиг фигуры без изменения формы);
  MC_<cls>: фигура -> круг 0,25 м в representative_point (микрокруг)."""
import sys, json, time, resource, pickle, gc
resource.setrlimit(resource.RLIMIT_AS, (3_800_000_000, 3_800_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np, shapely
from shapely.strtree import STRtree
from shapely import affinity
from collections import Counter
from scipy.spatial import cKDTree
from common import *
from expds.metric import ClassCurve, sweep_groups, curve_where
from expds import metric as M
mod = M.load_official(METRIC); calc = mod.F1Calculator()
T = pickle.load(open(W/'work/table.pkl', 'rb'))
CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}
cls, conf, reg, grp, cx, cy = T['cls'], T['conf'], T['reg'], T['grp'], T['cx'], T['cy']
KUR = ['kurgany_povrezhdennye', 'kurgany_tselye']
PL0 = {}
for (pc, gc_), (P, J, S) in T['pairs'].items():
    d = {}
    o = np.lexsort((J, P))  # порядок эталонов как в официальном цикле: при равном покрытии берётся меньший j
    for p, j, s in zip(P[o], J[o], S[o]): d.setdefault(int(p), []).append((int(j), float(s)))
    PL0[(pc, gc_)] = d
del T['pairs']; gc.collect()
GREG = {c: Counter(T['greg'][c]) for c in CL}
GTW = pickle.load(open(W/'work/gt_wkb.pkl', 'rb')); GT = {c: shapely.from_wkb(v) if len(v) else np.array([]) for c, v in GTW.items()}; del GTW
GTREE = {c: STRtree([x.buffer(calc.r_tolerance) if x.geom_type == 'Point' else x for x in GT[c]]) for c in CL if len(GT[c])}

def curves(PL):
    out = {}
    for c in CL:
        ii = np.flatnonzero(cls == cid[c]); o = ii[np.argsort(-conf[ii], kind='stable')]
        hit = np.zeros(len(o), bool); matched = set()
        for r, p in enumerate(o):
            lst = PL.get((CL[cls[p]], c), {}).get(int(p))
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r] = True
        out[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                            gt_regions=GREG[c], pred_group=list(grp[o]))
    return out

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
F10 = family_folds(regions, site_family, 10); FL = family_folds(regions, site_family, None)

t = time.time(); CV0 = curves(PL0)
taumin = {}
every = set().union(*F10)
for held in F10:
    res = sweep_groups({c: curve_where(v, every - held) for c, v in CV0.items()})
    for c, v in res.items():
        for g, gv in v['groups'].items():
            k = (g, c); taumin[k] = min(taumin.get(k, np.inf), gv['threshold'])
print('taumin', {f'{g}|{c}': round(v, 4) for (g, c), v in sorted(taumin.items())}, round(time.time()-t), 's', flush=True)
tau_c = np.array([taumin.get((str(grp[i]), CL[cls[i]]) if cls[i] >= 0 else ('', ''), np.inf) for i in range(len(cls))])
SEL = (cls >= 0) & (conf >= tau_c)
print('selected', int(SEL.sum()), Counter(CL[c] for c in cls[SEL]).most_common(), flush=True)
del tau_c
GW = pickle.load(open(W/'work/geom_wkb.pkl', 'rb'))
selidx = np.flatnonzero(SEL)
G = dict(zip(selidx.tolist(), shapely.from_wkb(GW[selidx]))); del GW; gc.collect()
print('geoms loaded', len(G), flush=True)

W8 = {'kurgany_povrezhdennye': 30, 'kurgany_tselye': 20, 'gorodishcha': 15}
def evaluate(cv):
    cnt, info = honest_counts(cv, F10)
    s, per = score_counts(cnt)
    return dict(all=s, new34=score_counts(cnt, new)[0], optical=score_counts(cnt, optical)[0],
                lofo=score_counts(honest_counts(cv, FL)[0])[0],
                rand=[score_counts(honest_counts(cv, rand_folds(sd))[0])[0] for sd in range(5)],
                per={c: per[c] for c in CL}, info=info), cnt

def repair(newgeom):
    """newgeom: {p: geometry}; returns PL with pairs of these p recomputed."""
    touched = {CL[cls[p]] for p in newgeom}
    PL = {k: (dict(v) if k[0] in touched else v) for k, v in PL0.items()}
    for p in newgeom:
        pc = CL[cls[p]]
        for gc_ in (KUR if pc in KUR else [pc]):
            PL.setdefault((pc, gc_), {}).pop(p, None)
    for p, g in newgeom.items():
        pc = CL[cls[p]]
        for gc_ in (KUR if pc in KUR else [pc]):
            if gc_ not in GTREE: continue
            lst = []
            for j in GTREE[gc_].query(g):
                ok, s = calc._match_quality(g, GT[gc_][j])
                if ok: lst.append((int(j), float(s)))
            if lst: PL[(pc, gc_)][p] = sorted(lst)
    return PL

def consensus(classes, r):
    ids = [cid[c] for c in classes]
    allm = np.isin(cls, ids)
    out = {}
    for rg in np.unique(reg[SEL & allm]):
        nb = np.flatnonzero(allm & (reg == rg)); tr = cKDTree(np.c_[cx[nb], cy[nb]])
        tg = np.flatnonzero(SEL & allm & (reg == rg))
        for p, lst in zip(tg, tr.query_ball_point(np.c_[cx[tg], cy[tg]], r)):
            q = nb[lst]; w = conf[q]
            dx = float((w * cx[q]).sum() / w.sum() - cx[p]); dy = float((w * cy[q]).sum() / w.sum() - cy[p])
            if abs(dx) + abs(dy) > 1e-6: out[int(p)] = affinity.translate(G[int(p)], dx, dy)
    return out
def micro(classes):
    ids = [cid[c] for c in classes]
    return {int(p): G[int(p)].representative_point().buffer(0.25, 16) for p in selidx if cls[p] in ids and G[int(p)].geom_type == 'Polygon'}

import os
CAND = '/workspace/koz3_shared/bot-6/work/localize/cand_li.npz'; SNAP = '/workspace/koz3_shared/bot-6/work/localize/snap_li.npz'
if not os.path.exists(SNAP):
    ii = np.array([p for p in selidx if CL[cls[p]] in KUR])
    lat = np.arctan(np.sinh(cy[ii] / 6378137.0))
    rad = np.sqrt(np.array([G[int(p)].area for p in ii]) / np.pi) * np.cos(lat)
    np.savez(CAND, idx=ii, reg=np.array([str(reg[p]) for p in ii]), grp=np.array([str(grp[p]) for p in ii]), cx=cx[ii], cy=cy[ii], rad=rad)
    print('dumped', len(ii), Counter(str(grp[p]) for p in ii).most_common()); sys.exit(0)
S = np.load(SNAP); SG = np.load(CAND)
def snap(v, groups=None):
    out = {}
    for k, p in enumerate(S['idx']):
        if not S['has'][k]: continue
        if groups is not None and str(SG['grp'][k]) not in groups: continue
        dx, dy = float(S['dx_' + v][k]), float(S['dy_' + v][k])
        if abs(dx) + abs(dy) > 1e-3: out[int(p)] = affinity.translate(G[int(p)], dx, dy)
    return out
LIG = {g for g in set(SG['grp'].tolist()) if g.startswith('li')}
print('li groups', LIG, flush=True)
VARS = [('A', None)]
for v in ('PK1', 'PK2', 'CM'):
    VARS.append((f'LI_{v}_li', lambda v=v: snap(v, LIG)))
    VARS.append((f'LI_{v}_all', lambda v=v: snap(v)))
only = set(sys.argv[1:])
RES = {}; CNT = {}
den = 100.0
out_path = '/workspace/koz3_shared/bot-6/work/localize/localize_li.json'
for name, fn in VARS:
    if only and name not in only and name != 'A': continue
    t = time.time()
    if fn is None: cv = CV0; nmoved = 0
    else:
        ng = fn(); nmoved = len(ng); PLn = repair(ng)
        sh = np.array([np.hypot(*(np.array(ng[p].centroid.coords[0]) - np.array(G[p].centroid.coords[0]))) for p in ng]) if ng else np.zeros(1)
        nchg = sum(1 for p in ng for gc_ in (KUR if CL[cls[p]] in KUR else [CL[cls[p]]]) if [j for j, _ in PLn.get((CL[cls[p]], gc_), {}).get(p, [])] != [j for j, _ in PL0.get((CL[cls[p]], gc_), {}).get(p, [])])
        print(name, 'shift3857 p50 %.3f p90 %.3f max %.3f; pairs changed (по составу эталонов) %d' % (np.median(sh), np.percentile(sh, 90), sh.max(), nchg), flush=True)
        cv = curves(PLn); del ng, PLn
    RES[name], CNT[name] = evaluate(cv); RES[name]['moved'] = nmoved
    if name != 'A':
        A, B = RES['A'], RES[name]
        bs = paired_bootstrap_counts(CNT[name], CNT['A'], site_family, 1000)
        d = B['all'] - A['all']
        p4 = {c: w * (B['per'][c]['f1'] - A['per'][c]['f1']) / den for c, w in W8.items()}
        B['vsA'] = dict(delta=d, d_new34=B['new34']-A['new34'], d_optical=B['optical']-A['optical'], d_lofo=B['lofo']-A['lofo'],
                        d_rand=[b-a for a, b in zip(A['rand'], B['rand'])], boot_B_minus_A=bs, P_A_gt_B=1-bs['p_gt0'], P4=p4,
                        rule={'P1': d >= 0.002, 'P2': bs['p_gt0'] >= 0.85,
                              'P3': (B['lofo']-A['lofo'] > 0) and sum(b-a > 0 for a, b in zip(A['rand'], B['rand'])) >= 4,
                              'P4': all(v >= -0.0005 for v in p4.values()) and B['new34'] >= A['new34']-0.001 and B['optical'] >= A['optical']-0.001,
                              'P5_package_only': d < 0.006})
        chg = {c: (round(B['per'][c]['tp']-A['per'][c]['tp']), round(B['per'][c]['fp']-A['per'][c]['fp'])) for c in CL if B['per'][c]['tp'] != A['per'][c]['tp']}
        print(name, 'moved', nmoved, 'delta %.6f new34 %+.6f opt %+.6f lofo %+.6f rand %s P(B>A) %.3f CI90 [%+.4f;%+.4f] dTP/dFP %s rule %s' % (
            d, B['vsA']['d_new34'], B['vsA']['d_optical'], B['vsA']['d_lofo'], [round(x, 5) for x in B['vsA']['d_rand']],
            bs['p_gt0'], bs['p5'], bs['p95'], chg, B['vsA']['rule']), round(time.time()-t), 's', flush=True)
    else:
        print('A', round(RES['A']['all'], 6), round(RES['A']['new34'], 6), round(RES['A']['optical'], 6), 'lofo', round(RES['A']['lofo'], 6), flush=True)
    json.dump(RES, open(out_path, 'w'), default=float, indent=1, ensure_ascii=False)
print('done')
