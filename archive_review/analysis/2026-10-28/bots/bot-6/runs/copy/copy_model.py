"""COPY: вторая копия оставленного предсказания (тот же полигон, тот же класс) там, где модель «объект виден ещё в одном слое /
есть второй эталон» даёт высокий балл. Кэш /workspace/rerank (stack_arch, база A). Копия = новый кандидат в группе copy_<cls>
с conf = балл модели; её порог подбирает sweep_groups на обучающих фолдах (как у всех групп).
Честность: внешние 10 фолдов семей; для внешнего k модель для фолда k обучена на 9 других; баллы копий в обучающих фолдах
(для подбора порога) — из внутренних моделей, не видевших ни k, ни свой фолд (полностью вложенно, 100 обучений).
П3 (LOFO, 5 случайных раскладок) — упрощённо: баллы OOF своей раскладки без внутренней вложенности (утечка только в порог)."""
import sys, json, time, resource, pickle, gc, re
resource.setrlimit(resource.RLIMIT_AS, (3_800_000_000, 3_800_000_000))
sys.path.insert(0, '/workspace/rerank')
import numpy as np, lightgbm as lgb
from collections import Counter, defaultdict
from scipy.spatial import cKDTree
from common import *
from expds.metric import ClassCurve, sweep_groups, curve_where
from expds.naming import modality_of
OUT = '/workspace/koz3_shared/bot-6/work/copy/'
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
N0 = len(cls)

def curves(extra=None):
    """extra: dict cls -> (orig_idx array, conf array, group name) — копии с парами оригинала."""
    out = {}
    for c in CL:
        ii = np.flatnonzero(cls == cid[c]); cc = conf[ii]; gg = grp[ii].astype(object); src = ii
        if extra and c in extra:
            ei, ec, eg = extra[c]
            src = np.r_[ii, ei]; cc = np.r_[cc, ec]; gg = np.r_[gg, np.array([eg] * len(ei), dtype=object)]
        o = np.argsort(-cc, kind='stable')
        hit = np.zeros(len(o), bool); matched = set()
        for r, k in enumerate(o):
            p = int(src[k]); lst = PL0.get((CL[cls[p]], c), {}).get(p)
            if not lst: continue
            best, bj = -np.inf, -1
            for j, s in lst:
                if j in matched: continue
                if s > best + 1e-9: best, bj = s, j
            if bj >= 0: matched.add(bj); hit[r] = True
        out[c] = ClassCurve(conf=cc[o].astype(np.float64), hit=hit, pred_region=list(reg[src[o]]), n_gt=T['n_gt'][c],
                            gt_regions=GREG[c], pred_group=list(gg[o]))
    return out

def fold_counts(cv, held, train, counts):
    res = sweep_groups({c: curve_where(v, train) for c, v in cv.items()})
    taus = {}
    for c, v in res.items():
        for g, gv in v['groups'].items(): taus.setdefault(g, {})[c] = gv['threshold']
    for c, v in cv.items():
        hc = curve_where(v, held); per = counts.setdefault(c, {})
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        if hc.conf.size == 0: continue
        tau = np.array([taus.get(g, {}).get(c, np.inf) for g in hc.pred_group])
        kept = hc.conf >= tau
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return taus

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
every = set().union(*F10)

# --- база A и taumin
t = time.time(); CV0 = curves()
cntA = {}; taumin = {}
for held in F10:
    taus = fold_counts(CV0, held, every - held, cntA)
    for g, d in taus.items():
        for c, v in d.items(): taumin[(g, c)] = min(taumin.get((g, c), np.inf), v)
print('A', score_counts(cntA)[0], round(time.time()-t), 's', flush=True)

# --- модальности участков (входные данные, s3_listing: растры/LAS, не разметка)
MODS = ['La', 'Li', 'Ae', 'SpOr', 'Or', 'MagGr', 'EtGr', 'TerFl']
site_mod = defaultdict(Counter)
for line in open('/workspace/data/s3_listing.txt', encoding='utf-8'):
    path = line.split()[-1] if line.strip() else ''
    if not path or path.endswith('/') or not re.search(r'\.(tif|tiff|las|laz|jpg|png)$', path, re.I): continue
    site = path.split('/')[0]; m = modality_of(path)
    if m: site_mod[site][m] += 1
miss = [r for r in regions if r not in site_mod]
print('regions without listing', len(miss), miss[:5], flush=True)

ELIG = ['kurgany_povrezhdennye', 'kurgany_tselye', 'finds_points', 'object_poly']
tau_c = np.array([taumin.get((str(grp[i]), CL[cls[i]]), np.inf) if cls[i] >= 0 else np.inf for i in range(N0)])
SEL = np.flatnonzero(np.isin(cls, [cid[c] for c in ELIG]) & (conf >= tau_c)); del tau_c
print('eligible', len(SEL), Counter(CL[c] for c in cls[SEL]).most_common(), flush=True)
# метка: у кандидата >= 2 эталонов своего класса в паре (второй эталон в той же точке)
y = np.array([len({j for j, _ in PL0.get((CL[cls[p]], CL[cls[p]]), {}).get(int(p), [])}) >= 2 for p in SEL], int)
print('label pos', int(y.sum()), Counter(CL[c] for c in cls[SEL][y == 1]), flush=True)
# признаки
GROUPS = sorted(set(grp[SEL])); gidx = {g: i for i, g in enumerate(GROUPS)}
def fam_of(g):
    g = str(g); return 'li' if g.startswith('li') else 'sp' if g.startswith('sp') else 'ae' if g.startswith('ae') else 'geo'
F = []
nb_cache = {}
for c in ELIG:
    ids = [cid[x] for x in (KUR if c in KUR else [c])]
    m_all = np.isin(cls, ids)
    for rg in np.unique(reg[SEL][cls[SEL] == cid[c]]):
        nb = np.flatnonzero(m_all & (reg == rg)); nb = nb[conf[nb] >= 0.02]
        tr = cKDTree(np.c_[cx[nb], cy[nb]]) if len(nb) else None
        tg = SEL[(cls[SEL] == cid[c]) & (reg[SEL] == rg)]
        for p in tg:
            f = {}
            lst = nb[tr.query_ball_point((cx[p], cy[p]), 3.0)] if tr is not None else np.zeros(0, int)
            lst = lst[lst != p]
            for fm in ('li', 'sp', 'ae', 'geo'):
                q = lst[np.array([fam_of(grp[x]) == fm for x in lst], bool)] if len(lst) else lst
                f[f'n_{fm}'] = len(q); f[f'max_{fm}'] = float(conf[q].max()) if len(q) else 0.0
                qs = q[cls[q] == cls[p]] if len(q) else q
                f[f'n_{fm}_same'] = len(qs)
            f['n_other_grp'] = int(np.sum(grp[lst] != grp[p])) if len(lst) else 0
            nb_cache[int(p)] = f
for p in SEL:
    f = nb_cache[int(p)]; sm = site_mod.get(str(reg[p]), Counter())
    row = [conf[p], np.log(conf[p] + 1e-6), gidx[grp[p]], cls[p], T['area'][p], T['bw'][p], T['bh'][p], T['nv'][p]]
    row += [f[k] for k in sorted(f)] + [sm.get(m, 0) for m in MODS] + [sum(1 for m in MODS if sm.get(m))]
    F.append(row)
FN = ['conf', 'logconf', 'grp', 'cls', 'area', 'bw', 'bh', 'nv'] + sorted(next(iter(nb_cache.values()))) + ['mod_' + m for m in MODS] + ['n_mods']
X = np.asarray(F, np.float64); del F, nb_cache
print('X', X.shape, flush=True)
freg = np.array([str(reg[p]) for p in SEL])
def fold_id(folds):
    m = {r: i for i, f in enumerate(folds) for r in f}; return np.array([m.get(r, -1) for r in freg])
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=1.0, num_threads=2, verbose=-1, seed=0)
def fit_pred(tr, te):
    if y[tr].sum() < 5: return np.zeros(te.sum())
    b = lgb.train(PAR, lgb.Dataset(X[tr], y[tr], categorical_feature=[2, 3]), 200)
    return b.predict(X[te])

def copies(score, thr_floor=0.01):
    ex = {}
    for c in ELIG:
        m = (cls[SEL] == cid[c]) & (score >= thr_floor)
        ex[c] = (SEL[m], score[m], 'copy')
    return ex

W8 = {'kurgany_povrezhdennye': 30, 'kurgany_tselye': 20, 'gorodishcha': 15}
def summary(name, cnt, extra_scores=None):
    s, per = score_counts(cnt); return dict(all=s, new34=score_counts(cnt, new)[0], optical=score_counts(cnt, optical)[0], per=per)
def compare(name, cntB, P3=None):
    A = summary('A', cntA); B = summary(name, cntB)
    bs = paired_bootstrap_counts(cntB, cntA, site_family, 1000); d = B['all'] - A['all']
    p4 = {c: w * (B['per'][c]['f1'] - A['per'][c]['f1']) / 100 for c, w in W8.items()}
    rule = {'P1': d >= 0.002, 'P2': bs['p_gt0'] >= 0.85, 'P3': P3,
            'P4': all(v >= -0.0005 for v in p4.values()) and B['new34'] >= A['new34'] - 0.001 and B['optical'] >= A['optical'] - 0.001,
            'P5_package_only': d < 0.006}
    chg = {c: (round(B['per'][c]['tp'] - A['per'][c]['tp']), round(B['per'][c]['fp'] - A['per'][c]['fp'])) for c in CL if B['per'][c]['tp'] != A['per'][c]['tp'] or B['per'][c]['fp'] != A['per'][c]['fp']}
    r = dict(A=A['all'], B=B['all'], delta=d, d_new34=B['new34'] - A['new34'], d_optical=B['optical'] - A['optical'], boot=bs,
             P_A_gt_B=1 - bs['p_gt0'], P4=p4, rule=rule, dTP_dFP=chg)
    print(name, json.dumps(r, default=float, ensure_ascii=False), flush=True)
    return r
RES = {}
def honest_with(score_for_outer):
    cnt = {}
    for k, held in enumerate(F10):
        cv = curves(copies(score_for_outer(k))); fold_counts(cv, held, every - held, cnt); del cv
    return cnt

# 1) оракул (метка известна) — потолок; 2) слепая копия (conf оригинала)
RES['oracle'] = compare('oracle', honest_with(lambda k: np.where(y == 1, conf[SEL], 0.0)))
RES['blind'] = compare('blind', honest_with(lambda k: conf[SEL].copy()))
# 3) модель, полностью вложенно
fid = fold_id(F10); K = len(F10)
t = time.time()
outer = np.zeros(len(SEL)); inner = {}
for k in range(K):
    outer[fid == k] = fit_pred(fid != k, fid == k)
    sc = np.zeros(len(SEL))
    for j in range(K):
        if j == k: continue
        sc[fid == j] = fit_pred((fid != k) & (fid != j), fid == j)
    sc[fid == k] = outer[fid == k]; inner[k] = sc
print('nested fits', round(time.time() - t), 's', flush=True)
from sklearn.metrics import roc_auc_score, average_precision_score
print('OOF AUC', roc_auc_score(y, outer), 'AP', average_precision_score(y, outer), 'pos rate', y.mean(), flush=True)
np.save(OUT + 'copy_oof.npy', outer)
cntM = honest_with(lambda k: inner[k])
# П3 упрощённо: LOFO и 5 раскладок на OOF-баллах своей раскладки
def oof_for(folds):
    f = fold_id(folds); sc = np.zeros(len(SEL))
    for k in range(len(folds)): sc[f == k] = fit_pred(f != k, f == k)
    return sc
p3 = []
for sd in range(5):
    fo = rand_folds(sd); sc = oof_for(fo); ev = set().union(*fo)
    cA, cB = {}, {}
    cvB = curves(copies(sc))
    for held in fo: fold_counts(CV0, held, ev - held, cA); fold_counts(cvB, held, ev - held, cB)
    p3.append(score_counts(cB)[0] - score_counts(cA)[0]); del cvB
    print('rand', sd, p3[-1], flush=True)
cvB = curves(copies(outer)); cA, cB = {}, {}
for held in FL: fold_counts(CV0, held, every - held, cA); fold_counts(cvB, held, every - held, cB)
lofo = score_counts(cB)[0] - score_counts(cA)[0]; del cvB
print('lofo', lofo, flush=True)
RES['model'] = compare('model', cntM, P3=bool(lofo > 0 and sum(x > 0 for x in p3) >= 4))
RES['model'].update(d_rand=p3, d_lofo=lofo, auc=roc_auc_score(y, outer), ap=average_precision_score(y, outer), features=FN)
b = lgb.train(PAR, lgb.Dataset(X, y, categorical_feature=[2, 3]), 200)
RES['model']['importance_gain'] = dict(sorted(zip(FN, b.feature_importance('gain').round(1).tolist()), key=lambda x: -x[1]))
json.dump(RES, open(OUT + 'copy_model.json', 'w'), default=float, indent=1, ensure_ascii=False)
print('done')
