"""bot-5. Курганы на Ae без лидара: классификатор кандидатов (виньетки 16–96 м, ручные признаки extract_ae.py,
БЕЗ предобученных весов), голова LightGBM 3 класса (фон / повреждённый / целый). Всё задано ЗАРАНЕЕ, не подбиралось:
LightGBM num_leaves 31, lr 0.05, 200 раундов, min_data_in_leaf 40, feature_fraction 0.8, bagging 0.8/1, 2 потока;
негативы обучения — случайные 4 % (вес 25), позитивы все; ignore — точки вне полигона ближе 2 м к эталону (только
из обучения). Точка-предсказание = круг r 0.5 м в (ux,uy) (как в сдаче). Добавление: строки с p_класса ≥ 0.2,
NMS 6 м внутри класса; группы kae_new (участок без лидара) / kae_new_li. Gate: conf_ae_строки × p(курган) кандидата
в её точке. Строгая вложенность: модель отложенного фолда обучена только на обучающих участках (весь фолд и его семьи
вне обучения); p обучающих участков для sweep_groups — внутренний OOF (5 семейных фолдов обучения).
Офиц. метрика: машинерия work/hillfort/gor_scene.py, обобщённая на 2 класса курганов; совпадение с honest_counts
проверяется assert. Режим FEATS=dino — признаки DINOv2-S (только замер, подмножество виньеток)."""
import sys, os, json, pickle, collections, time, math, glob
import numpy as np, shapely
from pathlib import Path
ROOT = Path('/workspace/koz3_shared/bot-5/repo'); sys.path.insert(0, str(ROOT))
from expds import metric as M
from expds.metric import ClassCurve, family_folds, honest_counts, score_counts, sweep_groups, curve_where, paired_bootstrap_counts
from expds.naming import site_family
M._SHM_CACHE_DIR = Path('/workspace/koz3_shared/bot-5/tmp/shm_off/x')
import lightgbm as lgb
OD = Path('/workspace/koz3_shared/bot-5/analysis/kurg_ae'); OD.mkdir(parents=True, exist_ok=True)
FD = Path(os.environ.get('FEATD', '/workspace/koz3_shared/bot-5/tmp/kurg_ae/feat'))
FDX = os.environ.get('FEATD_EXT')  # режим кандидатов bot-7: src2 из FEATD_EXT, из FEATD только строки стека (src1) [+src0 вне участков bot-7 при TRAIN_OWN=1 — только обучение]
GENSRC = 2 if FDX else 0; TRAIN_OWN = os.environ.get('TRAIN_OWN') == '1'
C7 = dict(np.load('/workspace/koz3_shared/bot-5/tmp/kurg_ae/cand7.npz')) if FDX else None  # dict: иначе npz читает массив заново при каждом обращении
C7REG = set(C7['reg'].tolist()) if FDX else set()
TAG = os.environ.get('TAG', 'hand'); FEATS = os.environ.get('FEATS', 'hand')
LAYOUTS = os.environ.get('LAYOUTS', 'fam10,LOFO,seed0,seed1,seed2,seed3,seed4').split(',')
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:.0f}s]', *a, flush=True)
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb'))
CL = T['CL']; cls, conf, reg, grp, mk = T['cls'], T['conf'], T['reg'], T['grp'], T['mk']
KC = ('kurgany_povrezhdennye', 'kurgany_tselye')
mod = M.load_official(str(ROOT / 'materials' / 'compute_metrics.py')); calc = mod.F1Calculator()
new = {l.strip() for l in (ROOT / 'notes' / 'regions_las_new.txt').read_text(encoding='utf-8').splitlines() if l.strip()}
boxes = set(json.loads((ROOT / 'notes' / 'las_boxes_59.json').read_text(encoding='utf-8')))
WK = pickle.load(open('/workspace/rerank/work/gt_wkb.pkl', 'rb'))
GG = {c: shapely.from_wkb(WK[c]) for c in KC}; GREG = {c: np.array(T['greg'][c]) for c in KC}
del WK
def greedy_hits(order_ids, pair_of):
    used = set(); hit = np.zeros(len(order_ids), bool)
    for n, i in enumerate(order_ids):
        bj, bs = -1, -np.inf
        for j, s in sorted(pair_of(i)):
            if j in used: continue
            if s > bs + 1e-9: bs, bj = s, j
        if bj >= 0: used.add(bj); hit[n] = True
    return hit
PAIRS = {}
def greedy_base(c):
    P, J, S = T['pairs'][(c, c)]; byc = collections.defaultdict(list)
    for a, b, s in zip(P, J, S): byc[int(a)].append((int(b), float(s)))
    if c in KC: PAIRS[c] = byc
    idx = np.flatnonzero(cls == CL.index(c)); o = idx[np.argsort(-conf[idx], kind='stable')]
    hit = greedy_hits(list(map(int, o)), lambda i: byc.get(i, []))
    return ClassCurve(conf=conf[o].astype(np.float64), hit=hit, pred_region=list(reg[o]), n_gt=T['n_gt'][c],
                      gt_regions=collections.Counter(T['greg'][c]), pred_group=list(grp[o]))
curves = {c: greedy_base(c) for c in CL}
regions = {r for c in curves.values() for r in c.pred_region} | {r for c in curves.values() for r in c.gt_regions}
optical = {r for r in regions if r} - boxes
folds10 = family_folds(regions, site_family, 10)
base_full, _ = honest_counts(curves, folds10)
A = score_counts(base_full)[0]; log('honest-10 A', A)
del curves
def rand_folds(seed, k=10):
    fams = {}
    for r in regions: fams.setdefault(site_family(r), set()).add(r)
    names = sorted(fams); lab = np.random.default_rng(seed).permutation(len(names)) % k
    return [set().union(*(fams[n] for n, l in zip(names, lab) if l == i)) for i in range(k)]
LAY = {'fam10': folds10, 'LOFO': family_folds(regions, site_family, None)}
for s in range(5): LAY[f'seed{s}'] = rand_folds(s)
def gsc(per):
    tp = sum(v[0] for v in per.values()); fp = sum(v[1] for v in per.values()); n = sum(v[2] for v in per.values())
    return dict(tp=tp, fp=fp, n_gt=n, f1=float(M._f1(tp, fp, n - tp)))
def auc(s, t):
    s = np.asarray(s, float); t = np.asarray(t, bool)
    if t.all() or (~t).all(): return float('nan')
    from scipy.stats import rankdata
    r = rankdata(s); return float((r[t].sum() - t.sum() * (t.sum() + 1) / 2) / (t.sum() * (~t).sum()))

# ---------- кандидаты ----------
recs, Fs, regs, gsds, wins, widx = [], [], [], [], [], []
WOFF = 0  # смещение строк виньеток (порядок = sorted(FEATD) затем sorted(FEATD_EXT)) — для dino
WROWS = []
for dd, kind in ([(FD, 'own')] + ([(Path(FDX), 'ext')] if FDX else [])):
    for f in sorted(dd.glob('*.npz')):
        z = np.load(f, allow_pickle=True)
        rec = z['rec']; zr = str(z['reg'])
        if not len(rec): continue
        wp = np.cumsum(z['widx']) - 1 + WOFF; WOFF += len(z['wins'])
        if FDX:
            if kind == 'own': m_ = (rec[:, 2] == 1) | ((rec[:, 2] == 0) & TRAIN_OWN & (zr not in C7REG))
            else: m_ = rec[:, 2] == 2
        else: m_ = np.ones(len(rec), bool)
        recs.append(rec[m_]); Fs.append(z['F'][m_].astype(np.float32)); regs += [zr] * int(m_.sum()); gsds += [float(z['gsd'])] * int(m_.sum())
        widx.append(z['widx'][m_]); WROWS.append(np.where(z['widx'], wp, -1)[m_])
        if FEATS == 'dino': wins.append(z['wins'])
REC = np.concatenate(recs); X = np.concatenate(Fs); CREG = np.array(regs, dtype=object); CGSD = np.array(gsds)
WIDX = np.concatenate(widx); WROW_ALL = np.concatenate(WROWS); WINS = True if FEATS == 'dino' else None; del wins
FEAT_NAMES = json.load(open(FD / 'feat_names.json'))
log('raw candidates', len(REC), collections.Counter(REC[:, 2].astype(int).tolist()))
# дедуп между перекрывающимися tif одного участка: предпочтение меньшему gsd
order = np.lexsort((CGSD, CREG.astype(str))); keep = np.zeros(len(REC), bool)
for r in sorted(set(CREG)):
    ii = order[CREG[order] == r]
    seen_stack = set(); kept_pts = []
    tree_pts = []
    for src in (1, 2, 0):
        jj = ii[REC[ii, 2] == src]
        if src in (1, 2):
            for j in jj:
                if (src, REC[j, 3]) in seen_stack: continue
                seen_stack.add((src, REC[j, 3])); keep[j] = True
            continue
        # gen: жадно по gsd, подавление ближе 3 м к уже взятому gen
        if not len(jj): continue
        g = np.argsort(CGSD[jj], kind='stable'); jj = jj[g]
        xy = REC[jj, :2]; from scipy.spatial import cKDTree
        tr = cKDTree(xy); taken = np.zeros(len(jj), bool); dead = np.zeros(len(jj), bool)
        for a in range(len(jj)):
            if dead[a]: continue
            taken[a] = True
            for b in tr.query_ball_point(xy[a], 3.0):
                if b != a and CGSD[jj[b]] > CGSD[jj[a]] - 1e-9 and not taken[b]: dead[b] = True
        keep[jj[taken]] = True
idx = np.flatnonzero(keep)
REC, X, CREG, CGSD, WIDX = REC[idx], X[idx], CREG[idx], CGSD[idx], WIDX[idx]
if WINS is not None:
    WROW = WROW_ALL[idx]
SRC = REC[:, 2].astype(int); SID = REC[:, 3].astype(int)
log('dedup candidates', len(REC), collections.Counter(SRC.tolist()))
# плотность пиков генератора (признак без меток): число gen-кандидатов в 30/60 м
from scipy.spatial import cKDTree
dens = np.zeros((len(REC), 3), np.float32)
for r in sorted(set(CREG)):
    ii = np.flatnonzero(CREG == r); gi = ii[SRC[ii] == (0 if not FDX or r not in C7REG else 2)]
    if not len(gi): continue
    tr = cKDTree(REC[gi, :2]); hiR = X[gi, 0] > np.percentile(X[gi, 0], 80)
    trh = cKDTree(REC[gi[hiR], :2]) if hiR.any() else None
    dens[ii, 0] = [len(v) for v in tr.query_ball_point(REC[ii, :2], 30.0)]
    dens[ii, 1] = [len(v) for v in tr.query_ball_point(REC[ii, :2], 60.0)]
    if trh is not None: dens[ii, 2] = [len(v) for v in trh.query_ball_point(REC[ii, :2], 60.0)]
X = np.c_[X, dens]; FEAT_NAMES = FEAT_NAMES + ['dens30', 'dens60', 'dens60_hiR']
if FDX:  # безметочные поля генератора bot-7 (score, радиус LoG, D, rank); для прочих строк NaN
    pos7 = {int(c): i for i, c in enumerate(C7['cand_id'])}; e7 = np.full((len(REC), 4), np.nan, np.float32)
    for k in np.flatnonzero(SRC == 2):
        i = pos7[int(SID[k])]; e7[k] = (C7['score'][i], C7['gen'][i], C7['D'][i], C7['rank'][i])
    X = np.c_[X, e7]; FEAT_NAMES = FEAT_NAMES + ['b7_score', 'b7_gen', 'b7_D', 'b7_rank']
# ---------- метки и пары круга 0.5 м ----------
R_CIRC = 0.5
PTS = shapely.points(REC[:, 0], REC[:, 1])
LAB = np.zeros(len(REC), np.int8); CPAIRS = {c: {} for c in KC}; IGN = np.zeros(len(REC), bool)
for ci, c in enumerate(KC):
    g = GG[c]; gr = GREG[c]
    for r in sorted(set(CREG)):
        ii = np.flatnonzero(CREG == r); jj = np.flatnonzero(gr == r)
        if not len(jj): continue
        tr = shapely.STRtree(g[jj])
        a, b = tr.query(PTS[ii], predicate='dwithin', distance=2.0)
        for p_, q_ in zip(a, b):
            k = ii[p_]; j = jj[q_]; gj = g[j]
            if shapely.within(PTS[k], gj):
                if LAB[k] == 0: LAB[k] = ci + 1
                ok, s = calc._match_quality(shapely.Point(REC[k, 0], REC[k, 1]).buffer(R_CIRC, 32), gj)
                if ok: CPAIRS[c].setdefault(int(k), []).append((int(j), float(s)))
            else:
                IGN[k] = True
IGN &= LAB == 0
np.savez(OD / f'diag_{TAG}.npz', LAB=LAB, SRC=SRC, SID=SID, CREG=CREG.astype(str), IGN=IGN, UX=REC[:, 0], UY=REC[:, 1])
log('labels', collections.Counter(LAB.tolist()), 'ignore', int(IGN.sum()), 'pairs', {c: len(v) for c, v in CPAIRS.items()})
# проверка: проходит ли круг 0.5 м / 2 м в representative_point эталона
chk = {}
for c in KC:
    for rc in (0.5, 2.0):
        ok = [calc._match_quality(g.representative_point().buffer(rc, 32), g)[0] for g in GG[c][:3000] if g.geom_type != 'Point']
        chk[f'{c}_r{rc}'] = float(np.mean(ok))
log('circle at GT representative_point hit-rate', chk)
# покрытие эталона кандидатами (верхняя граница recall генератора)
AE_REG = sorted(set(CREG))
cover = {}
for ci, c in enumerate(KC):
    gin = np.flatnonzero(np.isin(GREG[c], AE_REG)); hitj = {j for v in CPAIRS[c].values() for j, _ in v}
    hit_gen = {j for k, v in CPAIRS[c].items() if SRC[k] == GENSRC for j, _ in v}
    nolid = np.array([GREG[c][j] not in boxes for j in gin])
    cover[c] = dict(gt_in_ae_regions=int(len(gin)), any_cand=len(hitj & set(gin.tolist())), gen_cand=len(hit_gen & set(gin.tolist())),
                    gt_nolidar=int(nolid.sum()), any_cand_nolidar=len(hitj & set(gin[nolid].tolist())))
log('coverage', cover)
# ---------- признаки DINO (вариант 2) ----------
if FEATS == 'dino':
    D = np.load(os.environ.get('DINO_FILE', str(OD / 'dino_feats.npy'))).astype(np.float32)  # строки = sorted(FEATD) затем sorted(FEATD_EXT)
    sel = WROW >= 0; X = np.zeros((len(REC), D.shape[1]), np.float32); X[sel] = D[WROW[sel]]
    VALID = sel
else:
    VALID = np.ones(len(REC), bool)
X = np.where(np.isinf(X), np.nan, X).astype(np.float32)  # NaN оставляем LightGBM (missing)
PARAMS = dict(objective='multiclass', num_class=3, num_leaves=31, learning_rate=0.05, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, num_threads=2, verbose=-1, seed=0)
NROUND = 200; NEG_FRAC = 0.04
RNG_NEG = np.random.default_rng(1).random(len(REC)) < NEG_FRAC
TRAIN_SRC = os.environ.get('TRAIN_SRC')  # напр. '2': обучать только на кандидатах генератора (без строк стека: у них b7-поля NaN -> утечка «это строка стека»)
TRMASK = np.isin(SRC, [int(x) for x in TRAIN_SRC.split(',')]) if TRAIN_SRC else np.ones(len(REC), bool)
ADD_MIN = float(os.environ.get('ADD_MIN', 0.2))
def fit(ii):
    ii = ii[VALID[ii] & TRMASK[ii] & ~IGN[ii] & ((LAB[ii] > 0) | RNG_NEG[ii])]
    y = LAB[ii]; w = np.where(y > 0, 1.0, 1 / NEG_FRAC)
    if (y > 0).sum() < 5: return None
    return lgb.train(PARAMS, lgb.Dataset(X[ii], y, weight=w), NROUND)
def pred(m, ii):
    if m is None or not len(ii): return np.zeros((len(ii), 3), np.float32) + np.array([1, 0, 0], np.float32)
    return m.predict(X[ii]).astype(np.float32)
# ---------- кривые курганов (пересчёт жадного только на затронутых участках) ----------
TOUCH = set(AE_REG)
# замыкание: участки, чьи строки делят эталон (индекс GT в парах) с затронутыми, — иначе жадное сопоставление не разделимо
while True:
    g_t = {(c, j) for c in KC for i, v in PAIRS[c].items() if reg[i] in TOUCH for j, _ in v} | {(c, j) for c in KC for v in CPAIRS[c].values() for j, _ in v}
    add_ = {reg[i] for c in KC for i, v in PAIRS[c].items() if reg[i] not in TOUCH and any((c, j) in g_t for j, _ in v)}
    if not add_: break
    TOUCH |= add_
log('TOUCH regions', len(TOUCH), sorted(TOUCH - set(AE_REG)))
BASE = {}
for c in KC:
    rows = np.flatnonzero(cls == CL.index(c)); o = rows[np.argsort(-conf[rows], kind='stable')]
    hit = greedy_hits(list(map(int, o)), lambda i: PAIRS[c].get(i, []))
    t = np.isin(reg[o], list(TOUCH))
    BASE[c] = dict(fix_ids=o[~t], fix_conf=conf[o][~t].astype(float), fix_hit=hit[~t], fix_reg=reg[o][~t], fix_grp=grp[o][~t],
                   t_ids=o[t], t_conf=conf[o][t].astype(float), t_reg=reg[o][t], t_grp=grp[o][t], t_ae=(mk[o][t] == 'ae'))
# стековая ae-строка -> индекс кандидата
STK = {int(SID[k]): k for k in np.flatnonzero(SRC == 1)}
LIDAR_REG = boxes
GEN = np.flatnonzero(SRC == GENSRC)
NB6 = {}
for r in AE_REG:
    gi = GEN[CREG[GEN] == r]
    if len(gi):
        tr = cKDTree(REC[gi, :2])
        for a, v in zip(gi, tr.query_ball_point(REC[gi, :2], 6.0)): NB6[int(a)] = gi[v]
def build(c, ci, P_, mode):
    b = BASE[c]; ids = list(map(int, b['t_ids'])); cf = b['t_conf'].copy(); rg = list(b['t_reg']); gp = list(b['t_grp'])
    if 'gate' in mode:
        for n, i in enumerate(ids):
            if b['t_ae'][n] and i in STK:
                k = STK[i]
                if not np.isnan(P_[k, 0]): cf[n] = cf[n] * (1 - P_[k, 0])
    extra = {}; cf = list(cf)
    if 'add' in mode:
        pc = P_[GEN, ci + 1]; cand = GEN[(pc >= ADD_MIN) & ~np.isnan(pc)]
        cand = cand[np.argsort(-P_[cand, ci + 1], kind='stable')]; dead = set()
        nid = len(cls)
        for k in cand:
            if k in dead: continue
            for q in NB6.get(int(k), []):
                if q != k: dead.add(int(q))
            extra[nid] = CPAIRS[c].get(int(k), []); ids.append(nid); cf.append(float(P_[k, ci + 1])); rg.append(CREG[k])
            gp.append('kae_new_li' if CREG[k] in LIDAR_REG else 'kae_new'); nid += 1
    cf = np.array(cf); o = np.argsort(-cf, kind='stable'); ids_o = [ids[k] for k in o]
    hit = greedy_hits(ids_o, lambda i: extra[i] if i >= len(cls) else PAIRS[c].get(i, []))
    allc = np.r_[b['fix_conf'], cf[o]]; allh = np.r_[b['fix_hit'], hit]
    allr = list(b['fix_reg']) + [rg[k] for k in o]; allg = list(b['fix_grp']) + [gp[k] for k in o]
    allid = np.r_[b['fix_ids'], np.array(ids_o, dtype=np.int64)]
    oo = np.lexsort((allid, -allc))  # как argsort(-conf, stable) по номеру строки в greedy_base (связи conf≈1 у ae-строк!)
    return ClassCurve(conf=allc[oo], hit=allh[oo], pred_region=[allr[k] for k in oo], n_gt=T['n_gt'][c],
                      gt_regions=collections.Counter(T['greg'][c]), pred_group=[allg[k] for k in oo])
def fold_counts(c, ci, train, held, P_, mode, per):
    cv = build(c, ci, P_, mode) if mode != 'base' else build(c, ci, None, '')
    res = sweep_groups({c: curve_where(cv, train)}); taus = {g: gv['threshold'] for g, gv in res[c]['groups'].items()}
    hc = curve_where(cv, held)
    for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
    tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
    for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
def counts(c, ci, folds, P_list, mode):
    every = set().union(*folds); per = {}
    for i, held in enumerate(folds): fold_counts(c, ci, every - held, held, None, mode, per)
    return per
for ci, c in enumerate(KC):
    chk_ = counts(c, ci, folds10, None, 'base')
    bad = [(r, chk_.get(r), v) for r, v in base_full[c].items() if v != [0, 0, 0] and chk_.get(r, [0, 0, 0]) != v]
    if bad: log(c, 'MISMATCH', len(bad), bad[:15], 'tot own', gsc(chk_), 'tot hc', gsc(base_full[c]))
    assert all(chk_.get(r, [0, 0, 0]) == v for r, v in base_full[c].items() if v != [0, 0, 0]), f'{c} counts mismatch'
log('own kurgan routine == honest_counts: OK')
# ---------- модели по раскладкам ----------
def inner_oof(tr):
    P_ = np.full((len(REC), 3), np.nan, np.float32)
    regs_ = sorted(set(CREG[tr])); fo = family_folds(regs_, site_family, 5)
    for h in fo:
        te = np.isin(CREG[tr], list(h))
        if te.any(): P_[tr[te]] = pred(fit(tr[~te]), tr[te])
    return P_
ALL_AE = np.flatnonzero(VALID); P_ALL_INNER = None
def fold_probs(every, held):
    """p всех кандидатов для внешнего фолда: held — модель на обучении, обучение — внутренний OOF.
    Если в held нет кандидатов Ae, обучение = все кандидаты, p — общий внутренний OOF (held их не видит)."""
    global P_ALL_INNER
    he = np.flatnonzero(np.isin(CREG, list(held)) & VALID)
    if not len(he):
        if P_ALL_INNER is None: P_ALL_INNER = inner_oof(ALL_AE)
        return P_ALL_INNER, he, 0
    tr = np.flatnonzero(np.isin(CREG, list(every - held)) & VALID)
    P_ = inner_oof(tr); P_[he] = pred(fit(tr), he)
    return P_, he, 6
def cand_report(P_oof):
    ok = ~np.isnan(P_oof[:, 0]) & ~IGN; y = LAB[ok] > 0; p = 1 - P_oof[ok, 0]
    out = dict(auc_any=auc(p, y), n_pos=int(y.sum()), n_neg=int((~y).sum()))
    for ci, c in enumerate(KC):
        yy = LAB[ok] == ci + 1; out[f'auc_{c}'] = auc(P_oof[ok, ci + 1], yy)
    nl = ok & ~np.isin(CREG, list(boxes)); out['auc_any_nolidar'] = auc(1 - P_oof[nl, 0], LAB[nl] > 0)
    stk = ok & (SRC == 1); out['auc_any_stack_rows'] = auc(1 - P_oof[stk, 0], LAB[stk] > 0); out['n_stack_rows_pos'] = int((LAB[stk] > 0).sum()); out['n_stack_rows'] = int(stk.sum())
    gen = ok & (SRC == GENSRC); out['auc_any_gen'] = auc(1 - P_oof[gen, 0], LAB[gen] > 0)
    for ci, c in enumerate(KC): out[f'auc_gen_{c}'] = auc(P_oof[gen, ci + 1], LAB[gen] == ci + 1)
    out['n_gen_pos'] = int((LAB[gen] > 0).sum()); out['n_gen'] = int(gen.sum())
    # recall эталона: доля GT (участки с Ae) с кандидатом p>=t внутри; t фиксированы заранее 0.5 и top-1%
    for t in (0.5, 0.2):
        for ci, c in enumerate(KC):
            gin = set(np.flatnonzero(np.isin(GREG[c], AE_REG)).tolist())
            hj = {j for k, v in CPAIRS[c].items() if ok[k] and (1 - P_oof[k, 0]) >= t for j, _ in v}
            fpn = int(((1 - P_oof[:, 0]) >= t)[ok & (LAB == 0)].sum())
            out[f'gt_recall_p>={t}_{c}'] = [len(hj & gin), len(gin)]; out[f'neg_cands_p>={t}'] = fpn
    return out
RES = {'definition': __doc__, 'A_honest10': A, 'n_candidates': int(len(REC)), 'src_counts': dict(collections.Counter(SRC.tolist())),
       'labels': {str(k): v for k, v in collections.Counter(LAB.tolist()).items()}, 'ignore': int(IGN.sum()), 'coverage': cover,
       'circle_check': chk, 'feats': FEATS, 'train_src': TRAIN_SRC, 'add_min': ADD_MIN, 'n_valid': int(VALID.sum()), 'layouts': {}}
MODES = ['gate', 'add', 'add+gate'] if FEATS == 'hand' else ['gate']
base_c = {}
for nm in LAYOUTS:
    fo = LAY[nm]; every = set().union(*fo); P_oof = np.full((len(REC), 3), np.nan, np.float32); nfit = 0
    per = {(mode, c): {} for mode in MODES for c in KC}
    for c in KC: base_c[(nm, c)] = counts(c, KC.index(c), fo, None, 'base')
    for i, held in enumerate(fo):
        P_, he, nf = fold_probs(every, held); nfit += nf; P_oof[he] = P_[he]
        for mode in MODES:
            for ci, c in enumerate(KC): fold_counts(c, ci, every - held, held, P_, mode, per[(mode, c)])
    log(nm, 'fits', nfit)
    if nm in ('fam10', 'LOFO'):
        RES['layouts'].setdefault(nm, {})['cand'] = cand_report(P_oof); log(nm, 'cand', RES['layouts'][nm]['cand'])
        if nm == 'fam10' and FDX:
            f7 = dict(zip(C7['cand_id'].tolist(), C7['fold'].tolist())); agree = 0; sc = []; got = set()
            for k in np.flatnonzero((SRC == 2) & ~np.isnan(P_oof[:, 0])):
                myf = next(i for i, f in enumerate(fo) if CREG[k] in f); agree += myf == f7[int(SID[k])]
                sc.append(dict(cand_id=int(SID[k]), p=round(float(1 - P_oof[k, 0]), 5), p_povr=round(float(P_oof[k, 1]), 5),
                               p_tsel=round(float(P_oof[k, 2]), 5), fold=int(f7[int(SID[k])]), ok=True)); got.add(int(SID[k]))
            miss = [int(c) for c in C7['cand_id'] if int(c) not in got]
            sc += [dict(cand_id=c, p=None, p_povr=None, p_tsel=None, fold=int(f7[c]), ok=False) for c in miss]
            sc.sort(key=lambda d: d['cand_id'])
            RES['cand7_export'] = dict(n=len(sc), n_ok=len(got), n_missing=len(miss), fold_agree=[int(agree), len(got)])
            json.dump(sc, open(OD / f'scores_cand7{"" if TAG == "c7" else "_" + TAG}.json', 'w')); log('cand7 export', RES['cand7_export'])
        if nm == 'fam10' and FEATS == 'hand' and not FDX:
            sc = []
            for k in np.flatnonzero(~np.isnan(P_oof[:, 0])):
                sc.append(dict(cand_id=f'kae5_{"gen" if SRC[k]==0 else "stack"}_{CREG[k][:3]}_{k}', reg=CREG[k], ux=round(float(REC[k, 0]), 2), uy=round(float(REC[k, 1]), 2),
                               src=int(SRC[k]), stack_row=int(SID[k]) if SRC[k] == 1 else None, p=round(float(1 - P_oof[k, 0]), 5),
                               p_povr=round(float(P_oof[k, 1]), 5), p_tsel=round(float(P_oof[k, 2]), 5),
                               fold=next(i for i, f in enumerate(fo) if CREG[k] in f)))
            json.dump(sc, open(OD / f'scores_own_{TAG}.json', 'w'), ensure_ascii=False)
        if nm == 'fam10' and FEATS == 'hand':
            ii = np.flatnonzero(VALID & TRMASK & ~IGN & ((LAB > 0) | RNG_NEG))
            imp = lgb.train(PARAMS, lgb.Dataset(X[ii], LAB[ii], weight=np.where(LAB[ii] > 0, 1.0, 1 / NEG_FRAC)), NROUND).feature_importance('gain')
            RES['feat_gain_top20'] = sorted(zip(FEAT_NAMES, map(float, imp)), key=lambda x: -x[1])[:20]
        if nm == 'fam10' and FEATS == 'dino':
            np.save(OD / f'p_oof_fam10_{TAG}.npy', P_oof)
    if nm == 'fam10' and FEATS == 'hand': np.save(OD / f'p_oof_fam10_{TAG}.npy', P_oof)
    lo = RES['layouts'].setdefault(nm, {})
    for mode in MODES:
        pv = {c: per[(mode, c)] for c in KC}
        vf = dict(base_full); vf.update(pv); bf = dict(base_full); bf.update({c: base_c[(nm, c)] for c in KC})
        d = score_counts(vf)[0] - score_counts(bf)[0]
        e = dict(delta=d, per_class={c: [gsc(base_c[(nm, c)]), gsc(pv[c])] for c in KC})
        if nm == 'fam10':
            e.update(delta_exact=score_counts(vf)[0] - A, delta_new34=score_counts(vf, new)[0] - score_counts(base_full, new)[0],
                     delta_optical=score_counts(vf, optical)[0] - score_counts(base_full, optical)[0],
                     delta_class={c: score_counts(dict(base_full) | {c: pv[c]})[0] - A for c in KC},
                     bootstrap=paired_bootstrap_counts(vf, base_full, site_family, n=1000, seed=0))
        lo[mode] = e; log(nm, mode, 'delta', round(d, 6), {c: gsc(pv[c]) for c in KC})
    json.dump(RES, open(OD / f'kurg_eval_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
if all(n in RES['layouts'] for n in ('fam10', 'LOFO', 'seed0')):
    for mode in MODES:
        d = RES['layouts']['fam10'][mode]; seeds = [RES['layouts'][f'seed{s}'][mode]['delta'] for s in range(5) if f'seed{s}' in RES['layouts']]
        lofo = RES['layouts']['LOFO'][mode]['delta']
        RES.setdefault('acceptance', {})[mode] = {'P1': [d['delta_exact'], d['delta_exact'] >= 0.002], 'P2': [d['bootstrap'].get('p_gt0'), (d['bootstrap'].get('p_gt0') or 0) >= 0.85],
            'P3': [lofo, seeds, lofo > 0 and sum(x > 0 for x in seeds) >= 4],
            'P4': [d['delta_class'], d['delta_new34'], d['delta_optical'], min(d['delta_class'].values()) >= -0.0005 and d['delta_new34'] >= -0.001 and d['delta_optical'] >= -0.001],
            'P5_below_MDE': d['delta_exact'] < 0.006}
        log(mode, 'acceptance', RES['acceptance'][mode])
json.dump(RES, open(OD / f'kurg_eval_{TAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
import resource; log('done; peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
