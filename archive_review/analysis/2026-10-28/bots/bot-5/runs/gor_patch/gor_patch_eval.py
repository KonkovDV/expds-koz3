"""bot-5. Городища без лидара: классификатор виньеток SpOr 384 м (+контекст 768 м) на замороженных DINOv2-S
признаках (extract_feats.py), голова — L2-логрегрессия (numpy, PCA-64 внутри обучения, веса классов сбалансированы).
PCA считается без меток на обучающих виньетках внешнего фолда и общая для внутренних фитов. Гиперпараметры заданы ЗАРАНЕЕ, не подбирались: PCA 64, C=1, K=2 строки на сцену, R_CIRC 10 м, Landauer 0,85/0,05.
Метки виньетки: pos — ≥50 % площади эталона городища sp_or внутри виньетки; ignore — пересекает любое городище
(любая разметка) иначе; neg — остальное. Строгая вложенность: всё, что видит отложенный фолд, обучено только на
обучающих участках; p обучающих сцен (для порога sweep_groups) — из внутреннего OOF (5 семейных фолдов обучения).
Чистка Landauer 2025 (jcaa.178): OOF-p на обучающем наборе (5 семейных фолдов) -> убрать pos с p<0,85 и neg с p>0,05
-> переобучить. Точка: полигон = пересечение виньеток связной компоненты с p ≥ 0,5·pmax вокруг максимума,
polylabel (pole of inaccessibility) -> круг 10 м. Офиц. метрика — машинерия work/hillfort/gor_scene.py."""
import sys, json, math, collections, time
from pathlib import Path
import numpy as np
src = Path('/workspace/koz3_shared/bot-5/work/hillfort/gor_scene.py').read_text(encoding='utf-8').split('VARS = [')[0]
exec(compile(src, 'gor_scene_head', 'exec'))
from shapely.ops import polylabel, unary_union
from shapely.geometry import Polygon
OD = Path('/workspace/koz3_shared/bot-5/analysis/gor_patch')
import os
FT = os.environ.get('FEATS_TAG', '')  # '' = DINOv2-S с бокса; иначе feats{FT}.npz/meta{FT}.json (напр. с Kaggle)
F = np.load(OD / f'feats{FT}.npz')['F'].astype(np.float32); META = json.load(open(OD / f'meta{FT}.json', encoding='utf-8'))
PREG = np.array([m_['reg'] for m_ in META]); POLY = [Polygon(m_['corners']) for m_ in META]
log('patches', len(META), 'dim', F.shape[1], 'scenes', len(set(PREG)))
# ---- метки виньеток ----
GALL = GG; GSP = np.flatnonzero(gm == 'sp_or')
lab = np.zeros(len(META), np.int8); hf_of = [[] for _ in META]
for k, P in enumerate(POLY):
    r = PREG[k]; st = 0
    for j in np.flatnonzero(greg == r):
        g = GALL[j]; gg = g if g.geom_type != 'Point' else g.buffer(5)
        if not P.intersects(gg): continue
        if gm[j] == 'sp_or' and P.intersection(gg).area >= 0.5 * gg.area: st = 1; hf_of[k].append(int(j))
        elif st == 0: st = -1
    lab[k] = st
log('labels', collections.Counter(lab.tolist()), 'sp_or hillforts with ≥1 pos patch', len({j for h in hf_of for j in h}), 'of', len(GSP))
PC, CREG, CLEAN_HI, CLEAN_LO, K_ROWS = 64, 1.0, 0.85, 0.05, 2
QQ = None   # проекция PCA-64, считается на обучающих виньетках внешнего фолда (без меток) и общая для внутренних фитов
def set_proj(idx):
    global QQ
    Xt = F[idx]; mu = Xt.mean(0); sd = Xt.std(0) + 1e-6
    sub = idx[np.random.default_rng(0).permutation(len(idx))[:4000]]
    U, S, Vt = np.linalg.svd((F[sub] - mu) / sd, full_matrices=False); W = Vt[:PC].T
    Q = ((F - mu) / sd) @ W; Q /= Q[idx].std(0) + 1e-6; QQ = np.c_[np.ones(len(Q)), Q]
def fit(idx):
    yt = (lab[idx] == 1).astype(float)
    if yt.min() == yt.max(): return ('const', float(yt.mean()))
    Q = QQ[idx]; sw = np.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
    w = np.zeros(Q.shape[1]); lam = np.r_[0, np.ones(Q.shape[1] - 1)] / CREG
    for _ in range(50):
        p = 1 / (1 + np.exp(-Q @ w)); gr = Q.T @ (sw * (p - yt)) + lam * w
        H = (Q * (sw * p * (1 - p))[:, None]).T @ Q + np.diag(lam) + 1e-9 * np.eye(len(w)); st = np.linalg.solve(H, gr); w -= st
        if np.abs(st).max() < 1e-8: break
    return ('lr', w)
def pred(mdl, idx):
    if mdl[0] == 'const': return np.full(len(idx), mdl[1])
    return 1 / (1 + np.exp(-QQ[idx] @ mdl[1]))
CLEAN_LOG = collections.Counter()
def inner_oof(idx, nf=5):
    regs = sorted(set(PREG[idx])); fo = family_folds(regs, site_family, nf); p = np.zeros(len(idx))
    for h in fo:
        te = np.isin(PREG[idx], list(h)); 
        if te.any(): p[te] = pred(fit(idx[~te]), idx[te])
    return p
def fit_clean(idx, clean):
    li = idx[lab[idx] >= 0]
    if not clean: return fit(li), 0
    p = inner_oof(li); yl = lab[li] == 1
    keep = ~((yl & (p < CLEAN_HI)) | (~yl & (p > CLEAN_LO)))
    CLEAN_LOG['fits'] += 1; CLEAN_LOG['pos_removed'] += int((yl & ~keep).sum()); CLEAN_LOG['pos_total'] += int(yl.sum()); CLEAN_LOG['neg_removed'] += int((~yl & ~keep).sum()); CLEAN_LOG['neg_total'] += int((~yl).sum())
    if keep.sum() < 50 or (lab[li[keep]] == 1).sum() < 5: CLEAN_LOG['fallback'] += 1; return fit(li), 0   # заранее: если чистка выкидывает почти всё — без чистки
    return fit(li[keep]), int((~keep).sum())
import shapely as _sh
OVL = {}; NEAR = {}
for r in sorted(set(PREG)):
    ii = np.flatnonzero(PREG == r); polys = [POLY[k] for k in ii]; tree = _sh.STRtree(polys)
    for a, k in enumerate(ii):
        OVL[int(k)] = {int(ii[b]) for b in tree.query(polys[a]) if polys[a].intersection(polys[b]).area > 0}
        NEAR[int(k)] = {int(ii[b]) for b in tree.query(polys[a].buffer(192))}
def decode(r, idx, p):
    """строки сцены: до K_ROWS компонент; conf = pmax компоненты; точка = polylabel пересечения виньеток компоненты."""
    rows = []; pos = {int(k): n for n, k in enumerate(idx)}; alive = np.ones(len(idx), bool)
    for _ in range(K_ROWS):
        if not alive.any(): break
        a = int(np.argmax(np.where(alive, p, -1))); pm = p[a]; comp = {a}; fr = [a]
        while fr:
            u = fr.pop()
            for kk in OVL[int(idx[u])]:
                v = pos.get(kk)
                if v is not None and v not in comp and alive[v] and p[v] >= 0.5 * pm: comp.add(v); fr.append(v)
        inter = POLY[idx[a]]
        for v in sorted(comp, key=lambda v: -p[v]):
            t = inter.intersection(POLY[idx[v]])
            if t.area > 0.05 * POLY[idx[a]].area: inter = t
        if inter.geom_type != 'Polygon': inter = max(getattr(inter, 'geoms', [POLY[idx[a]]]), key=lambda g: g.area)
        pt = polylabel(inter, tolerance=1.0)
        rows.append((r, float(pm), circle_pairs(r, pt.x, pt.y), (pt.x, pt.y)))
        for kk in NEAR[int(idx[a])]:
            v = pos.get(kk)
            if v is not None: alive[v] = False   # подавление окрестности 192 м
    return rows
def build_curve_rows(rows, gate_p=None):
    ids = list(map(int, gor_rows)); cf = list(conf[gor_rows].astype(float)); rg = list(reg[gor_rows]); gp = list(grp[gor_rows])
    if gate_p is not None:
        for n, i in enumerate(ids):
            if mk[i] == 'sp_or' and rg[n] in gate_p: cf[n] = cf[n] * gate_p[rg[n]]
    extra = {}
    for k, (r, c, pr, _) in enumerate(rows):
        nid = N0 + k; extra[nid] = pr; ids.append(nid); cf.append(c); rg.append(r); gp.append('gor_patch')
    cf = np.array(cf); o = np.argsort(-cf, kind='stable'); ids = [ids[k] for k in o]
    hit = greedy_hits(ids, lambda i: extra[i] if i >= N0 else pairs_by.get(i, []))
    return ClassCurve(conf=cf[o], hit=hit, pred_region=[rg[k] for k in o], n_gt=T['n_gt'][G],
                      gt_regions=collections.Counter(T['greg'][G]), pred_group=[gp[k] for k in o])
ALLIDX = np.arange(len(META))
def run_layout(folds, clean, collect=None):
    """возвращает per-вариант gor counts + OOF p виньеток отложенных фолдов."""
    every = set().union(*folds); per = {'add': {}, 'gate': {}, 'add+gate': {}}; p_out = np.full(len(META), np.nan); thr_out = {}
    removed = []
    for held in folds:
        train = every - held
        tri = ALLIDX[np.isin(PREG, list(train))]; hei = ALLIDX[np.isin(PREG, list(held))]
        if len(tri): set_proj(tri)
        rows_tr, rows_he, gp_ = [], [], {}
        if len(tri):
            # p обучающих сцен — внутренний OOF (каждая внутренняя модель тоже с чисткой на своём обучении)
            regs = sorted(set(PREG[tri])); fo = family_folds(regs, site_family, 5); ptr = np.zeros(len(tri))
            for h in fo:
                te = np.isin(PREG[tri], list(h))
                if te.any(): ptr[te] = pred(fit_clean(tri[~te], clean)[0], tri[te])
            mdl, nrem = fit_clean(tri, clean); removed.append(nrem)
            phe = pred(mdl, hei) if len(hei) else np.zeros(0); p_out[hei] = phe
            # порог виньетки: max F1 на внутреннем OOF обучения (для отчёта recall)
            yl = lab[tri] == 1; ok = lab[tri] >= 0; best = (0, 0.5)
            for t in np.unique(np.round(ptr[ok], 3)):
                tp_ = (yl & ok & (ptr >= t)).sum(); fp_ = (~yl & ok & (ptr >= t)).sum(); fn_ = (yl & ok & (ptr < t)).sum()
                f1 = 2 * tp_ / max(2 * tp_ + fp_ + fn_, 1); best = max(best, (f1, t))
            for k in hei: thr_out[int(k)] = best[1]
            for idx_set, pp, rows in ((tri, ptr, rows_tr), (hei, phe, rows_he)):
                for r in sorted(set(PREG[idx_set]) & set(SCN)):
                    mm = PREG[idx_set] == r; rows += decode(r, idx_set[mm], pp[mm]); gp_[r] = float(pp[mm].max())
        for vn in per:
            rows = rows_tr + rows_he if 'add' in vn else []
            cv = build_curve_rows(rows, gp_ if 'gate' in vn else None)
            res = sweep_groups({G: curve_where(cv, train)}); taus = {g: gv['threshold'] for g, gv in res[G]['groups'].items()}
            hc = curve_where(cv, held)
            for r, n in hc.gt_regions.items(): per[vn].setdefault(r, [0, 0, 0])[2] += int(n)
            tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per[vn].setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        if collect is not None: collect += [rw for rw in rows_he]
    return per, p_out, thr_out, removed
def patch_report(p, thr):
    ok = (lab >= 0) & ~np.isnan(p); yy = lab[ok] == 1; pp = p[ok]
    win = []; 
    for r in set(PREG[ok & (lab == 1)]):
        mm = ok & (PREG == r)
        if 0 < (lab[mm] == 1).sum() < mm.sum(): win.append(auc(p[mm], lab[mm] == 1))
    t = np.array([thr[int(k)] for k in np.flatnonzero(ok)]); hit = pp >= t
    hf = {j for k in np.flatnonzero(ok & (lab == 1)) if p[k] >= thr[int(k)] for j in hf_of[k]}
    scen = sorted(set(PREG[ok]) & set(SCN)); smax = [np.nanmax(p[PREG == r]) for r in scen]; sy = [y[sidx[r]] for r in scen]
    return dict(patch_auc=auc(pp, yy), within_pos_scene_auc_mean=float(np.mean(win)), n_pos_scenes=len(win),
                patch_recall=float(hit[yy].mean()), patch_precision=float(hit[yy].sum() / max(hit.sum(), 1)), n_pos=int(yy.sum()), n_neg=int((~yy).sum()),
                fp_patches=int((hit & ~yy).sum()), hillfort_recall=len(hf) / len(GSP), scene_auc_maxp=auc(smax, sy))
R = {'definition': 'see docstring; EXPERIMENTS.md:13-20', 'A_honest10': A, 'n_patches': len(META),
     'labels': {str(k): v for k, v in collections.Counter(lab.tolist()).items()}, 'variants': {}}
base_g = {nm: gor_counts(f, None) for nm, f in layouts.items()}
for clean in (True, False):
    cname = 'clean' if clean else 'raw'; out = {}
    for nm, fo in layouts.items():
        per, p, thr, rem = run_layout(fo, clean)
        if nm in ('fam10', 'LOFO'): out.setdefault('patch', {})[nm] = patch_report(p, thr) | {'removed_by_cleaning_mean': float(np.mean(rem))}; log(cname, nm, out['patch'][nm])
        for vn, pv in per.items():
            b, v = gsc(base_g[nm]), gsc(pv); o_ = out.setdefault(vn, {}); o_[nm] = dict(base=b, var=v, delta=15 * (v['f1'] - b['f1']) / 100)
            if nm == 'fam10':
                vf = dict(base_full); vf[G] = pv
                o_[nm].update(delta_exact=score_counts(vf)[0] - A, delta_new34=score_counts(vf, new)[0] - score_counts(base_full, new)[0],
                              delta_optical=score_counts(vf, optical)[0] - score_counts(base_full, optical)[0],
                              bootstrap=paired_bootstrap_counts(vf, base_full, site_family, n=1000, seed=0))
            log(cname, vn, nm, round(o_[nm]['delta'], 6), v)
    for vn in ('add', 'gate', 'add+gate'):
        o_ = out[vn]; d = o_['fam10']; seeds = [o_[f'seed{s}']['delta'] for s in range(5)]
        o_['acceptance'] = {'P1': [d['delta_exact'], d['delta_exact'] >= 0.002], 'P2': [d['bootstrap'].get('p_gt0'), (d['bootstrap'].get('p_gt0') or 0) >= 0.85],
            'P3': [o_['LOFO']['delta'], seeds, o_['LOFO']['delta'] > 0 and sum(x > 0 for x in seeds) >= 4],
            'P4': [d['delta_new34'], d['delta_optical'], d['delta_exact'] >= -0.0005 and d['delta_new34'] >= -0.001 and d['delta_optical'] >= -0.001],
            'P5_below_MDE': d['delta_exact'] < 0.006}
        log(cname, vn, 'acceptance', o_['acceptance'])
    out['clean_log'] = dict(CLEAN_LOG); CLEAN_LOG.clear(); log(cname, 'clean_log', out['clean_log'])
    R['variants'][cname] = out
    json.dump(R, open(OD / f'gor_patch_eval{FT}.json', 'w'), ensure_ascii=False, indent=1, default=float)
log('done'); import resource; log('peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
