"""bot-5. Диагностика OOF AUC 0,435 классификатора сцены (work/hillfort/gor_scene.py).
Переиспользует признаки gor_scene.py (строки 1–97 исполняются как есть), затем:
(а) утечка/сдвиг семей: AUC при fam10, LOFO, случайных фолдах по участкам; AUC константы «доля позитивов в train»;
(б) масштаб: размер городища vs размер сцены; (в) метки: негативные сцены с городищем в другой разметке."""
import sys, json, math, collections
from pathlib import Path
src = Path('/workspace/koz3_shared/bot-5/work/hillfort/gor_scene.py').read_text(encoding='utf-8').split('# ---------- строки gor')[0]
exec(compile(src, 'gor_scene_head', 'exec'))
OUTD = Path('/workspace/koz3_shared/bot-5/analysis/gor_patch'); OUTD.mkdir(parents=True, exist_ok=True)
R = {}
fam = np.array([site_family(r) for r in SCN])
fc = collections.defaultdict(lambda: [0, 0])
for f_, yy in zip(fam, y): fc[f_][0] += 1; fc[f_][1] += int(yy)
pure = {f_: v for f_, v in fc.items()}
nf = len(fc); n_pure = sum(1 for v in fc.values() if v[1] in (0, v[0]))
sc_in_pure = sum(v[0] for v in fc.values() if v[1] in (0, v[0]))
R['families'] = {'n_families': nf, 'pure_label_families': n_pure, 'scenes_in_pure_families': sc_in_pure, 'n_scenes': len(SCN),
                 'top': sorted(([k, v[0], v[1]] for k, v in fc.items()), key=lambda t: -t[1])[:15]}
def oof(folds):
    p = np.zeros(len(SCN)); prior = np.zeros(len(SCN))
    for held in folds:
        te = np.array([sidx[r] for r in SCN if r in held]); tr = np.array([k for k in range(len(SCN)) if SCN[k] not in held])
        if not len(te): continue
        m = lr_fit(X[tr], y[tr]); p[te] = lr_pred(m, X[te]); prior[te] = y[tr].mean()
    return p, prior
lay = {'fam10': family_folds(SCN, site_family, 10), 'LOFO': family_folds(SCN, site_family, None)}
rng = np.random.default_rng(0)
for s in range(3):
    perm = rng.permutation(len(SCN)); lay[f'random_region10_s{s}'] = [set(SCN[k] for k in perm[i::10]) for i in range(10)]
R['oof_auc'] = {}
for k, fo in lay.items():
    p, prior = oof(fo)
    # AUC внутри смешанных семей (позитивы и негативы одной семьи) — без вклада межсемейного сдвига
    within = []
    for f_ in fc:
        m_ = fam == f_
        if 0 < y[m_].sum() < m_.sum():
            within.append(auc(p[m_], y[m_]))
    R['oof_auc'][k] = {'lr': auc(p, y), 'const_train_prior': auc(prior, y), 'within_mixed_families_mean': float(np.mean(within)) if within else None, 'n_mixed': len(within)}
    log(k, R['oof_auc'][k])
# одиночные признаки с наибольшим AUC в LOFO через одномерную LR
R['univariate_lofo'] = {}
for j, nm in enumerate(FN):
    p = np.zeros(len(SCN))
    for held in lay['LOFO']:
        te = np.array([sidx[r] for r in SCN if r in held]); tr = np.array([k for k in range(len(SCN)) if SCN[k] not in held])
        m = lr_fit(X[tr][:, [j]], y[tr]); p[te] = lr_pred(m, X[te][:, [j]])
    R['univariate_lofo'][nm] = auc(p, y)
# (б) масштаб: размеры эталонов городищ sp_or и сцен
sz = []; fr = []
for j in np.flatnonzero(gm == 'sp_or'):
    r = greg[j]
    if r not in centre: continue
    g = GG[j]; x0, y0, x1, y1 = g.bounds; d = max(x1 - x0, y1 - y0); sz.append(d)
    w, h = centre[r][2], centre[r][3]; fr.append(g.area / (w * h) if g.geom_type != 'Point' else 0.0)
R['scale'] = {'gor_sp_or_n': len(sz), 'gor_diam_m_q10_50_90': np.percentile(sz, [10, 50, 90]).tolist(),
              'scene_w_m_q10_50_90': np.percentile([centre[r][2] for r in SCN], [10, 50, 90]).tolist(),
              'gor_area_frac_of_scene_q10_50_90': np.percentile(fr, [10, 50, 90]).tolist(),
              'gor_per_pos_scene': collections.Counter(int(((greg == r) & (gm == 'sp_or')).sum()) for r in SCN if y[sidx[r]]).most_common()}
# (в) метки
oth = collections.Counter(); neg_with_other = []
for r in SCN:
    if y[sidx[r]] == 0:
        mt = sorted(set(gm[greg == r]))
        if mt: neg_with_other.append([r, mt])
R['labels'] = {'neg_scenes': int((y == 0).sum()), 'neg_scenes_with_gor_in_other_markup': neg_with_other,
               'pos_scenes_with_zero_gor_candidates': [r for r in SCN if y[sidx[r]] and X[sidx[r], FN.index('g_n')] == 0]}
json.dump(R, open(OUTD / 'diag_auc435.json', 'w'), ensure_ascii=False, indent=1, default=float)
log('done', json.dumps({k: R[k] for k in ('families', 'oof_auc', 'scale')}, ensure_ascii=False, default=float)[:3000])
log('labels', len(neg_with_other), 'neg with other markup;', len(R['labels']['pos_scenes_with_zero_gor_candidates']), 'pos w/o gor cand')
log('univariate LOFO', {k: round(v, 3) for k, v in R['univariate_lofo'].items()})
# (г) поправка на сдвиг приора фолда: logit(p) - logit(доля позитивов в train) — использует только train
def lg(v): v = np.clip(v, 1e-6, 1 - 1e-6); return np.log(v / (1 - v))
for k in ('fam10', 'LOFO'):
    p, prior = oof(lay[k]); R['oof_auc'][k]['lr_prior_corrected'] = auc(lg(p) - lg(prior), y); log(k, 'prior-corrected AUC', R['oof_auc'][k]['lr_prior_corrected'])
# (д) есть ли у сцены файл разметки городищ SpOr (пустой или нет)
D = Path('/workspace/data/train-expds-3'); mf = collections.Counter()
for r in SCN:
    fs = list((D / r).glob('*разметка*/*SpOr_городища*.geojson'))
    n = sum(len(json.load(open(f, encoding='utf-8')).get('features', [])) for f in fs)
    mf[(int(y[sidx[r]]), 'file' if fs else 'nofile', 'nonempty' if n else 'empty')] += 1
R['labels']['markup_file_presence'] = {str(k): v for k, v in mf.items()}; log('markup files', dict(mf))
json.dump(R, open(OUTD / 'diag_auc435.json', 'w'), ensure_ascii=False, indent=1, default=float)
