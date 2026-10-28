"""Честный перемер post.kurgan_cross_modal_twins на кэше /workspace/rerank (база A = stack_arch, 0.314615).
Варианты: A (база, функция не применяется — как в configs/default.toml),
D = только dedup_ae_vs_li_m=3.0, F = полная функция (dedup + ae_twin + sp_focal_twin), T = только близнецы без dedup.
Функция применяется к кандидатам ДО порогов (в pipeline — после): приближение, см. TYPES.md."""
import sys, json, time, resource, pickle
resource.setrlimit(resource.RLIMIT_AS, (3_500_000_000, 3_500_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
from common import *          # family_folds, honest_counts, score_counts, paired_bootstrap_counts, site_family, sets, load
from expds.metric import ClassCurve
T = pickle.load(open(W/'work/table.pkl', 'rb'))
CL = T['CL']; cid = {c: i for i, c in enumerate(CL)}
POVR, TSEL = cid['kurgany_povrezhdennye'], cid['kurgany_tselye']
cls, conf, reg, grp, cx, cy = T['cls'], T['conf'], T['reg'], T['grp'], T['cx'], T['cy']
print('groups povr', Counter(grp[cls == POVR]).most_common(), '\ngroups tsel', Counter(grp[cls == TSEL]).most_common(), flush=True)
PL = {}
for (pc, gc), (P, J, S) in T['pairs'].items():
    if gc not in ('kurgany_povrezhdennye', 'kurgany_tselye') and pc != gc: continue
    d = {}
    o = np.lexsort((J, P))  # порядок эталонов как в официальном цикле: при равном покрытии берётся меньший j
    for p, j, s in zip(P[o], J[o], S[o]): d.setdefault(int(p), []).append((int(j), float(s)))
    PL[(pc, gc)] = d
GREG = {c: Counter(T['greg'][c]) for c in CL}

def trees(mask):
    out = {}
    idx = np.flatnonzero(mask)
    for r in np.unique(reg[idx]):
        ii = idx[reg[idx] == r]; out[r] = cKDTree(np.c_[cx[ii], cy[ii]])
    return out
li_t = trees((cls == POVR) & (grp == 'li') & (conf >= 0.06415))
ae_t = trees((cls == POVR) & (grp == 'ae'))
sp_t = trees((cls == POVR) & (grp == 'sp_focal') & (conf >= 0.05))
def near(t, i, r):
    tr = t.get(reg[i]); return tr is not None and len(tr.query_ball_point((cx[i], cy[i]), r)) > 0
ae_povr = np.flatnonzero((cls == POVR) & (grp == 'ae'))
drop = np.zeros(len(cls), bool)
for i in ae_povr:
    if near(li_t, i, 3.0): drop[i] = True
ae_tw, sp_tw = [], []
for i in np.flatnonzero(cls == TSEL):
    if grp[i] == 'ae':
        if near(ae_t, i, 0.5) or near(li_t, i, 4.0): continue
        ae_tw.append(i)
    elif grp[i] == 'sp_focal':
        if near(sp_t, i, 2.0): continue
        sp_tw.append(i)
print('ae povr', len(ae_povr), 'dropped by dedup', int(drop.sum()), 'ae_twin', len(ae_tw), 'sp_focal_twin', len(sp_tw), flush=True)

def curves(keep, twins):
    """twins: list of (orig_idx, group_name) added as povr candidates."""
    out = {}
    for c in CL:
        m = (cls == cid[c]) & keep
        base = np.flatnonzero(m)
        items = [(int(p), str(grp[p])) for p in base]
        if c == 'kurgany_povrezhdennye': items += twins
        ii = np.array([p for p, _ in items], np.int64); gg = [g for _, g in items]
        o = np.argsort(-conf[ii], kind='stable')
        hit = np.zeros(len(o), bool); matched = set()
        for r, k in enumerate(o):
            p = int(ii[k]); lst = PL.get((CL[cls[p]], c), {}).get(p)
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r] = True
        out[c] = ClassCurve(conf=conf[ii][o].astype(np.float64), hit=hit, pred_region=list(reg[ii][o]),
                            n_gt=T['n_gt'][c], gt_regions=GREG[c], pred_group=[gg[k] for k in o])
    return out

gt_regions = set().union(*[set(GREG[c]) for c in CL])
regions = {r for i in range(len(CL)) for r in np.unique(reg[cls == i])} | gt_regions  # как common.honest: участки кривых
boxes = json.loads((ROOT/'notes'/'las_boxes_59.json').read_text(encoding='utf-8'))
optical = {r for r in gt_regions if r} - set(boxes)
new = {l.strip() for l in (ROOT/'notes'/'regions_las_new.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
fams = sorted({site_family(r) for r in regions})
def rand_folds(seed, k=10):
    rng = np.random.default_rng(seed); f = list(fams); rng.shuffle(f)
    b = {x: i % k for i, x in enumerate(f)}; out = [set() for _ in range(k)]
    for r in regions: out[b[site_family(r)]].add(r)
    return [x for x in out if x]
F10 = family_folds(regions, site_family, 10)
FL = family_folds(regions, site_family, None)
W8 = {'kurgany_povrezhdennye': 30, 'kurgany_tselye': 20, 'gorodishcha': 15}
VAR = {'A': (np.ones(len(cls), bool), []), 'D': (~drop, []),
       'F': (~drop, [(i, 'ae_twin') for i in ae_tw] + [(i, 'sp_focal_twin') for i in sp_tw]),
       'T': (np.ones(len(cls), bool), [(i, 'ae_twin') for i in ae_tw] + [(i, 'sp_focal_twin') for i in sp_tw])}
res = {}; CNT = {}
for name, (keep, tw) in VAR.items():
    t = time.time(); cv = curves(keep, tw)
    cnt, info = honest_counts(cv, F10)
    s_all, per = score_counts(cnt); s_new = score_counts(cnt, new)[0]; s_opt = score_counts(cnt, optical)[0]
    lofo = score_counts(honest_counts(cv, FL)[0])[0]
    rs = [score_counts(honest_counts(cv, rand_folds(sd))[0])[0] for sd in range(5)]
    CNT[name] = cnt
    res[name] = dict(all=s_all, new34=s_new, optical=s_opt, lofo=lofo, rand=rs, info=info,
                     per={c: per[c] for c in W8})
    print(name, round(s_all, 6), round(s_new, 6), round(s_opt, 6), 'lofo', round(lofo, 6), 'rand', [round(x, 6) for x in rs],
          'povr', {k: round(v, 4) for k, v in per['kurgany_povrezhdennye'].items()}, info, round(time.time()-t), 's', flush=True)
den = sum(v for c, v in {'kurgany_povrezhdennye':30,'kurgany_tselye':20,'gorodishcha':15,'arkhitektury':10,'object_poly':10,'finds_points':10,'fortifikatsii':5}.items())
A = res['A']
for name in ('D', 'F', 'T'):
    B = res[name]
    bs = paired_bootstrap_counts(CNT[name], CNT['A'], site_family, 1000)
    bs_ab = paired_bootstrap_counts(CNT['A'], CNT[name], site_family, 1000)
    d = B['all'] - A['all']
    p4 = {c: w * (B['per'][c]['f1'] - A['per'][c]['f1']) / den for c, w in W8.items()}
    rule = {'P1_delta>=0.002': d >= 0.002, 'P2_P(B>A)>=0.85': bs['p_gt0'] >= 0.85,
            'P3_lofo>0_and_rand>=4/5': (B['lofo'] - A['lofo'] > 0) and sum(b - a > 0 for a, b in zip(A['rand'], B['rand'])) >= 4,
            'P4': all(v >= -0.0005 for v in p4.values()) and B['new34'] >= A['new34'] - 0.001 and B['optical'] >= A['optical'] - 0.001,
            'P5_package_only(delta<0.006)': d < 0.006}
    res[name]['vsA'] = dict(delta=d, d_new34=B['new34']-A['new34'], d_optical=B['optical']-A['optical'],
                            d_lofo=B['lofo']-A['lofo'], d_rand=[b-a for a, b in zip(A['rand'], B['rand'])],
                            boot_B_minus_A=bs, P_A_gt_B=bs_ab['p_gt0'], P4_weighted=p4, rule=rule)
    print(name, json.dumps(res[name]['vsA'], default=float), flush=True)
json.dump(dict(res=res, n=dict(ae_povr=len(ae_povr), dropped=int(drop.sum()), ae_twin=len(ae_tw), sp_twin=len(sp_tw))),
          open('/workspace/koz3_shared/bot-6/work/twins/twins_honest.json', 'w'), default=float, indent=1, ensure_ascii=False)
print('done')
