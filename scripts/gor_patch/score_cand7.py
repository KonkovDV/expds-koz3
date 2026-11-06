""". ТОЛЬКО ЧЕСТНЫЙ ЗАМЕР. OOF-вероятность городища для кандидатов (cands_gor_oof_D50.json).
Модель = та же, что в gor_patch_eval.py (DINOv2-S [CLS,mean] 384+768 м, PCA-64 на обучении, logreg C=1, чистка Landauer 0,85/0,05).
Для фолда f (поле fold кандидата, honest-10): обучение на виньетках участков, у которых fold кандидатов != f И семья
(site_family) не встречается среди участков фолда f; предсказание — виньетки участков фолда f.
p кандидата = p ближайшей по центру виньетки той же сцены (reg) с |dx|,|dy| ≤ 192 м (как frag_bot5); p_cov_max — max по таким.
Кандидаты без такой виньетки (нет SpOr-покрытия: Ae/Or-участки без SpOr, края) — не выдаются (счёт в логе).
Признаки считаются по SpOr-снимку; для kind Ae/Or это та же точка на SpOr-снимке, если он есть."""
import os, sys, json, collections, time
from pathlib import Path
import numpy as np
src = Path(os.environ.get('GOR_PATCH_EVAL', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gor_patch_eval.py'))).read_text(encoding='utf-8').split('import shapely as _sh')[0]
exec(compile(src, 'gor_patch_eval_head', 'exec'))
CF = os.environ.get('CANDS', '/workspace/koz3_shared/bot-7/work/gor/cands_gor_oof_D50.json')
C = json.load(open(CF, encoding='utf-8')); log('cands', len(C), collections.Counter(c['kind'] for c in C))
# --- согласованность фолдов: поле fold vs folds10 gor_scene
cfold = {}
for c in C:
    if cfold.setdefault(c['reg'], c['fold']) != c['fold']: raise SystemExit(f"region {c['reg']} has 2 folds")
f10 = {r: i for i, s in enumerate(folds10) for r in s}
common = [r for r in cfold if r in f10]
pairs = collections.Counter((cfold[r], f10[r]) for r in common)
m = {}; ok = True
for (a, b), n in pairs.items():
    if m.setdefault(a, b) != b: ok = False
log('fold check: regions with cand fold', len(cfold), 'in folds10', len(common), 'same partition (bijection)', ok and len(set(m.values())) == len(m), dict(pairs))
famf = collections.defaultdict(set)
for r, f in cfold.items(): famf[site_family(r)].add(f)
split_fams = {fa: sorted(fs) for fa, fs in famf.items() if len(fs) > 1}
log('families split across cand folds', split_fams)
PUX = np.array([m_['ux'] for m_ in META]); PUY = np.array([m_['uy'] for m_ in META])
p_patch = np.full(len(META), np.nan); pfold = {}
for f in sorted(set(cfold.values())):
    held = {r for r, ff in cfold.items() if ff == f}; hfam = {site_family(r) for r in held}
    tri = ALLIDX_ = np.flatnonzero(np.array([(cfold.get(r, -99) != f) and (site_family(r) not in hfam) for r in PREG]))
    hei = np.flatnonzero(np.isin(PREG, list(held)))
    if not len(hei): continue
    set_proj(tri); mdl, nrem = fit_clean(tri, True)
    p_patch[hei] = pred(mdl, hei)
    log('fold', f, 'held regs', len(held), 'held patches', len(hei), 'train patches', len(tri), 'train pos', int((lab[tri] == 1).sum()), 'removed', nrem)
# --- кандидат -> виньетка
byreg = collections.defaultdict(list)
for k, r in enumerate(PREG): byreg[r].append(k)
byreg = {r: np.array(v) for r, v in byreg.items()}
out = []; miss = collections.Counter(); agree = collections.Counter()
for c in C:
    ii = byreg.get(c['reg'])
    if ii is None: miss[(c['kind'], 'no_patches_in_reg')] += 1; continue
    dx = np.abs(PUX[ii] - c['ux']); dy = np.abs(PUY[ii] - c['uy']); w = (dx <= 192) & (dy <= 192)
    if not w.any(): miss[(c['kind'], 'no_patch_within_192')] += 1; continue
    jj = ii[w]; k = int(jj[np.argmin(np.hypot(dx[w], dy[w]))])
    if np.isnan(p_patch[k]): miss[(c['kind'], 'patch_not_predicted')] += 1; continue
    fb = c.get('frag_bot5', -1)
    if fb is not None and fb >= 0: agree['frag_same' if fb == k else 'frag_diff'] += 1
    out.append({'cand_id': int(c['cand_id']), 'p': float(p_patch[k]), 'fold': int(c['fold']), 'p_cov_max': float(np.nanmax(p_patch[jj])), 'patch': k, 'kind': c['kind']})
log('scored', len(out), 'by kind', collections.Counter(o['kind'] for o in out), 'missing', dict(miss), 'frag_bot5 agreement', dict(agree))
# --- честная проверка: AUC p среди кандидатов (метка = центр внутри эталона городища любой разметки той же сцены)
from shapely.geometry import Point
hitc = []
gidx = collections.defaultdict(list)
for j in range(len(GALL)): gidx[greg[j]].append(j)
cid = {int(c['cand_id']): c for c in C}
for o in out:
    c = cid[o['cand_id']]; P_ = Point(c['ux'], c['uy'])
    hitc.append(any((GALL[j] if GALL[j].geom_type != 'Point' else GALL[j].buffer(5)).contains(P_) for j in gidx.get(c['reg'], [])))
hitc = np.array(hitc); pp = np.array([o['p'] for o in out]); pm = np.array([o['p_cov_max'] for o in out])
rep = {'n_scored': len(out), 'n_hit': int(hitc.sum()), 'auc_p': auc(pp, hitc), 'auc_p_cov_max': auc(pm, hitc)}
for kd in ('SpOr', 'Ae', 'Or'):
    mk_ = np.array([o['kind'] == kd for o in out])
    if mk_.any() and 0 < hitc[mk_].sum() < mk_.sum(): rep[f'auc_p_{kd}'] = auc(pp[mk_], hitc[mk_]); rep[f'n_{kd}'] = int(mk_.sum()); rep[f'hit_{kd}'] = int(hitc[mk_].sum())
sc = np.array([cid[o['cand_id']]['score'] for o in out], float); gen = [cid[o['cand_id']]['gen'] for o in out]
for g in sorted(set(gen)):
    mg = np.array([x == g for x in gen])
    if 0 < hitc[mg].sum() < mg.sum(): rep[f'gen:{g}'] = dict(n=int(mg.sum()), hit=int(hitc[mg].sum()), auc_p=auc(pp[mg], hitc[mg]), auc_score=auc(sc[mg], hitc[mg]))
log('check', rep)
json.dump([{'cand_id': o['cand_id'], 'p': o['p'], 'fold': o['fold'], 'p_cov_max': o['p_cov_max'], 'patch': o['patch']} for o in out],
          open(OD / 'scores_cand7.json', 'w'))
json.dump({'definition': __doc__, 'cands_file': CF, 'missing': {f'{a}|{b}': n for (a, b), n in miss.items()}, 'fold_pairs': {f'{a}->{b}': n for (a, b), n in pairs.items()},
           'split_families': split_fams, 'frag_agreement': dict(agree), 'check': rep, 'clean_log': dict(CLEAN_LOG)},
          open(OD / 'scores_cand7_report.json', 'w'), ensure_ascii=False, indent=1, default=float)
log('done')
