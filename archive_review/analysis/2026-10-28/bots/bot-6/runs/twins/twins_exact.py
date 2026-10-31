"""Точная честная оценка kurgan_cross_modal_twins ПОСЛЕ порогов (порядок watchdog.finish: пороги -> twins).
Фолд k: пороги групп = sweep_groups по кривым БАЗЫ на остальных фолдах (twins не влияют на пороги, как в сдаче);
на удержанных участках оставляем conf >= порога (группы без порога выбрасываются, unseen=drop), затем пост-шаг,
затем официальное жадное сопоставление (пары _match_quality из кэша) только по оставленным.
Варианты: A — без пост-шага; TOae / TOsp — только ae_twin / только sp_focal_twin; TO — только близнецы (ae_twin + sp_focal_twin, dedup_ae_vs_li_m=0); F — вся функция.
merge_within_markup не моделируется (одинаково для всех вариантов)."""
import sys, json, time, resource, pickle, gc
resource.setrlimit(resource.RLIMIT_AS, (3_500_000_000, 3_500_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
from common import *
from expds.metric import ClassCurve, sweep_groups, curve_where
OUT = '/workspace/koz3_shared/bot-6/work/twins/'
T = pickle.load(open(W/'work/table.pkl', 'rb'))
CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}
cls, conf, reg, grp, cx, cy = T['cls'], T['conf'], T['reg'], T['grp'], T['cx'], T['cy']
POVR, TSEL = cid['kurgany_povrezhdennye'], cid['kurgany_tselye']
PL0 = {}
for (pc, gc_), (P, J, S) in T['pairs'].items():
    d = {}; o = np.lexsort((J, P))
    for p, j, s in zip(P[o], J[o], S[o]): d.setdefault(int(p), []).append((int(j), float(s)))
    PL0[(pc, gc_)] = d
del T['pairs']; gc.collect()
GREG = {c: Counter(T['greg'][c]) for c in CL}
grp_s = grp.astype(str)

def curves_base():
    out = {}
    for c in CL:
        ii = np.flatnonzero(cls == cid[c]); o = ii[np.argsort(-conf[ii], kind='stable')]
        hit = match(o, c)
        out[c] = ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                            gt_regions=GREG[c], pred_group=list(grp[o]))
    return out
def match(order_idx, c):
    hit = np.zeros(len(order_idx), bool); matched = set()
    for r, p in enumerate(order_idx):
        lst = PL0.get((CL[cls[p]], c), {}).get(int(p))
        if not lst: continue
        best, bj = -np.inf, -1
        for j, s in lst:
            if j in matched: continue
            if s > best + 1e-9: best, bj = s, j
        if bj >= 0: matched.add(bj); hit[r] = True
    return hit

def tree(idx):
    out = {}
    for r in np.unique(reg[idx]):
        ii = idx[reg[idx] == r]; out[r] = cKDTree(np.c_[cx[ii], cy[ii]])
    return out
def near(t, i, r):
    tr = t.get(reg[i]); return tr is not None and len(tr.query_ball_point((cx[i], cy[i]), r)) > 0

def post_twins(kept, dedup, twins):
    """kept: индексы после порогов. Возвращает (оставленные индексы, список индексов-копий в povr, stats)."""
    kp = kept[cls[kept] == POVR]
    li_t = tree(kp[(grp_s[kp] == 'li') & (conf[kp] >= 0.06415)])
    ae_t = tree(kp[grp_s[kp] == 'ae'])
    sp_t = tree(kp[(grp_s[kp] == 'sp_focal') & (conf[kp] >= 0.05)])
    keep = kept
    nd = 0
    if dedup:
        drop = np.array([cls[i] == POVR and grp_s[i] == 'ae' and near(li_t, i, 3.0) for i in kept], bool)
        keep = kept[~drop]; nd = int(drop.sum())
    tw = []; st = Counter()
    if twins:
        for i in kept[cls[kept] == TSEL]:
            g = grp_s[i]
            if g == 'ae' and twins in (True, 'ae'):
                if near(ae_t, i, 0.5) or near(li_t, i, 4.0): continue
                tw.append(i); st['ae_twin'] += 1
            elif g == 'sp_focal' and twins in (True, 'sp'):
                if near(sp_t, i, 2.0): continue
                tw.append(i); st['sp_focal_twin'] += 1
    st['dedup_dropped'] = nd
    return keep, np.array(tw, np.int64), st

def eval_folds(CV, folds, variants):
    every = set().union(*folds)
    cnt = {v: {} for v in variants}; stats = {v: Counter() for v in variants}
    regarr = reg.astype(object)
    for held in folds:
        res = sweep_groups({c: curve_where(v, every - held) for c, v in CV.items()})
        tau = {}
        for c, v in res.items():
            for g, gv in v['groups'].items(): tau[(g, cid[c])] = gv['threshold']
        hm = np.isin(regarr, list(held)) & (cls >= 0)
        hi = np.flatnonzero(hm)
        th = np.array([tau.get((grp_s[i], int(cls[i])), np.inf) for i in hi])
        kept = hi[conf[hi] >= th]
        for v, (dd, tw) in variants.items():
            keep, twi, st = post_twins(kept, dd, tw); stats[v].update(st)
            for c in CL:
                per = cnt[v].setdefault(c, {})
                for r in held:
                    n = GREG[c].get(r, 0)
                    if n: per.setdefault(r, [0, 0, 0])[2] += int(n)
                ii = keep[cls[keep] == cid[c]]; cc = conf[ii]; src = ii
                if c == 'kurgany_povrezhdennye' and len(twi):
                    src = np.r_[ii, twi]; cc = np.r_[cc, conf[twi]]
                o = np.argsort(-cc, kind='stable'); order = src[o]
                hit = match(order, c)
                for p, h in zip(order, hit): per.setdefault(reg[p], [0, 0, 0])[0 if h else 1] += 1
    return cnt, stats

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
VARS = {'A': (False, False), 'TO': (False, True), 'TOae': (False, 'ae'), 'TOsp': (False, 'sp'), 'F': (True, True), 'D': (True, False)}
t = time.time(); CV = curves_base(); print('curves', round(time.time() - t), 's', flush=True)
R = {}
t = time.time(); C10, S10 = eval_folds(CV, F10, VARS)
for v in VARS:
    s, per = score_counts(C10[v])
    R[v] = dict(all=s, new34=score_counts(C10[v], new)[0], optical=score_counts(C10[v], optical)[0],
                per={c: per[c] for c in ('kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha')}, stats=dict(S10[v]))
    print(v, round(s, 6), round(R[v]['new34'], 6), round(R[v]['optical'], 6), R[v]['per']['kurgany_povrezhdennye'], dict(S10[v]), flush=True)
print('k10', round(time.time() - t), 's', flush=True)
CL_ = {}
for name, folds in [('lofo', FL)] + [(f'rand{sd}', rand_folds(sd)) for sd in range(5)]:
    c, _ = eval_folds(CV, folds, VARS)
    for v in VARS: R[v][name] = score_counts(c[v])[0]
    print(name, {v: round(R[v][name], 6) for v in VARS}, flush=True)
W8 = {'kurgany_povrezhdennye': 30, 'kurgany_tselye': 20, 'gorodishcha': 15}
for v in ('TO', 'TOae', 'TOsp', 'F', 'D'):
    A, B = R['A'], R[v]
    bs = paired_bootstrap_counts(C10[v], C10['A'], site_family, 1000)
    d = B['all'] - A['all']
    p4 = {c: w * (B['per'][c]['f1'] - A['per'][c]['f1']) / 100 for c, w in W8.items()}
    drand = [B[f'rand{s}'] - A[f'rand{s}'] for s in range(5)]
    rule = {'P1': d >= 0.002, 'P2': bs['p_gt0'] >= 0.85, 'P3': (B['lofo'] - A['lofo'] > 0) and sum(x > 0 for x in drand) >= 4,
            'P4': all(x >= -0.0005 for x in p4.values()) and B['new34'] >= A['new34'] - 0.001 and B['optical'] >= A['optical'] - 0.001,
            'P5_package_only': d < 0.006}
    R[v]['vsA'] = dict(delta=d, d_new34=B['new34'] - A['new34'], d_optical=B['optical'] - A['optical'], d_lofo=B['lofo'] - A['lofo'],
                       d_rand=drand, boot=bs, P_A_gt_B=1 - bs['p_gt0'], P4=p4, rule=rule)
    print(v, json.dumps(R[v]['vsA'], default=float), flush=True)
json.dump(R, open(OUT + 'twins_exact.json', 'w'), default=float, indent=1, ensure_ascii=False)
print('done')
