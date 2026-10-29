# bot-5. ТОЛЬКО ЧЕСТНЫЙ ЗАМЕР (не для сдачи). Плотный локализатор городищ на спутнике: офиц. метрика на боксе.
"""Шаг 3 (только бокс): пики dense_probe.py (peaks{DTAG}.json, с бокса или с Kaggle) -> строки городищ «круг 10 м» ->
машинерия work/hillfort/gor_scene.py (база A, honest-10, LOFO, seed0-4, П1-П5). Сценовый гейт — виньеточный v3
(gor_patch_eval.py, чистка Landauer, строгая вложенность, те же фолды). conf точки = среднее(p гейта сцены, пик карты)
(задано заранее). Пороги — sweep_groups только на обучающих участках фолда (новая группа 'gor_dense').
Варианты: 'gate' (v3 без точек, контроль), '<var>:add' (только точки), '<var>:add+gate' (точки + гейт v3).
Env: DOUT, DTAG, OUT (папка виньеток gor_patch для гейта)."""
import os, sys, json, math, collections, time
from pathlib import Path
import numpy as np
os.environ.setdefault('OUT', '/workspace/koz3_shared/bot-5/analysis/gor_patch')
DOUT = Path(os.environ.get('DOUT', '/workspace/koz3_shared/bot-5/analysis/gor_dense')); DTAG = os.environ.get('DTAG', '')
_src = (Path(__file__).resolve().parents[1] / 'gor_patch' / 'gor_patch_eval.py').read_text(encoding='utf-8').split("R = {'definition'")[0]
exec(compile(_src, 'gor_patch_eval_head', 'exec'))
GCACHE = {}
PK = json.load(open(DOUT / f'peaks{DTAG}.json', encoding='utf-8')); SCNS = set(SCN)
VARS_D = [k for k in PK['layouts'][next(iter(PK['layouts']))][0] if k not in ('held', 'cp_s')]
log('peaks', PK['model'], PK['gsds'], 'layouts', list(PK['layouts']), 'variants', VARS_D)
def build_rows(rows, gate_p):
    ids = list(map(int, gor_rows)); cf = list(conf[gor_rows].astype(float)); rg = list(reg[gor_rows]); gp = list(grp[gor_rows])
    if gate_p is not None:
        for n, i in enumerate(ids):
            if mk[i] == 'sp_or' and rg[n] in gate_p: cf[n] = cf[n] * gate_p[rg[n]]
    extra = {}
    for k, (r, c, pr) in enumerate(rows):
        nid = N0 + k; extra[nid] = pr; ids.append(nid); cf.append(c); rg.append(r); gp.append('gor_dense')
    cf = np.array(cf); o = np.argsort(-cf, kind='stable'); ids = [ids[k] for k in o]
    hit = greedy_hits(ids, lambda i: extra[i] if i >= N0 else pairs_by.get(i, []))
    return ClassCurve(conf=cf[o], hit=hit, pred_region=[rg[k] for k in o], n_gt=T['n_gt'][G],
                      gt_regions=collections.Counter(T['greg'][G]), pred_group=[gp[k] for k in o])
def run(nm):
    folds = layouts[nm]; every = set().union(*folds); per = {k: {} for k in ['gate'] + [f'{v}:{w}' for v in VARS_D for w in ('add', 'add+gate')]}
    assert len(folds) == len(PK['layouts'][nm]) and all(set(a['held']) == set(b) for a, b in zip(PK['layouts'][nm], folds)), 'fold mismatch'
    for fr, held in zip(PK['layouts'][nm], folds):
        train = every - held; tri = ALLIDX[np.isin(PREG, list(train))]; hei = ALLIDX[np.isin(PREG, list(held))]; gp_ = {}
        key = frozenset(PREG[tri].tolist())
        if not len(hei) and key in GCACHE: gp_ = dict(GCACHE[key])   # отложенный фолд без виньеток: гейт тот же
        elif len(tri):
            set_proj(tri); regs = sorted(set(PREG[tri])); fo = family_folds(regs, site_family, 5); ptr = np.zeros(len(tri))
            for h in fo:
                te = np.isin(PREG[tri], list(h))
                if te.any(): ptr[te] = pred(fit_clean(tri[~te], True)[0], tri[te])
            mdl, _ = fit_clean(tri, True); phe = pred(mdl, hei) if len(hei) else np.zeros(0)
            for idx_set, pp in ((tri, ptr), (hei, phe)):
                for r in sorted(set(PREG[idx_set]) & SCNS): gp_[r] = float(pp[PREG[idx_set] == r].max())
            if not len(hei): GCACHE[key] = dict(gp_)
        rows = {v: [(r, 0.5 * (gp_.get(r, 0.0) + ps), circle_pairs(r, x, yy)) for r, pk in fr.get(v, {}).items() if r in SCNS for ps, x, yy in pk] for v in VARS_D}
        for vn in per:
            if vn == 'gate': cv = build_rows([], gp_)
            else:
                v, w = vn.split(':'); cv = build_rows(rows[v], gp_ if 'gate' in w else None)
            res = sweep_groups({G: curve_where(cv, train)}); taus = {g: gv['threshold'] for g, gv in res[G]['groups'].items()}
            hc = curve_where(cv, held)
            for r, n in hc.gt_regions.items(): per[vn].setdefault(r, [0, 0, 0])[2] += int(n)
            tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per[vn].setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    return per
R = {'definition': 'docstrings scripts/gor_dense/{dense_probe,gor_dense_eval}.py; EXPERIMENTS.md:13-20', 'A_honest10': A, 'peaks_file': f'peaks{DTAG}.json',
     'model': PK['model'], 'gsds': PK['gsds'], 'centre_hit': PK['centre_hit'], 'probe_stats': PK['stats'], 'layouts': {}}
for nm in PK['layouts']:
    per = run(nm); b = gsc(gor_counts(layouts[nm], None)); out = {}
    for vn, pv in per.items():
        vv = gsc(pv); o_ = dict(base=b, var=vv, delta=15 * (vv['f1'] - b['f1']) / 100)
        if nm == 'fam10':
            vf = dict(base_full); vf[G] = pv
            o_.update(delta_exact=score_counts(vf)[0] - A, delta_new34=score_counts(vf, new)[0] - score_counts(base_full, new)[0],
                      delta_optical=score_counts(vf, optical)[0] - score_counts(base_full, optical)[0],
                      bootstrap=paired_bootstrap_counts(vf, base_full, site_family, n=1000, seed=0))
        out[vn] = o_; log(nm, vn, round(o_['delta'], 6), vv, o_.get('delta_exact'), o_.get('delta_optical'))
    R['layouts'][nm] = out; json.dump(R, open(DOUT / f'gor_dense_eval{DTAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
if 'fam10' in R['layouts']:
    acc = {}
    for vn, d in R['layouts']['fam10'].items():
        seeds = [R['layouts'][f'seed{s}'][vn]['delta'] for s in range(5) if f'seed{s}' in R['layouts']]
        lofo = R['layouts'].get('LOFO', {}).get(vn, {}).get('delta')
        acc[vn] = {'P1': [d['delta_exact'], d['delta_exact'] >= 0.002], 'P2': [d['bootstrap'].get('p_gt0'), (d['bootstrap'].get('p_gt0') or 0) >= 0.85],
                   'P3': [lofo, seeds, lofo is not None and lofo > 0 and len(seeds) == 5 and sum(x > 0 for x in seeds) >= 4],
                   'P4': [d['delta_new34'], d['delta_optical'], d['delta_exact'] >= -0.0005 and d['delta_new34'] >= -0.001 and d['delta_optical'] >= -0.001],
                   'P5_below_MDE': d['delta_exact'] < 0.006}
        log('acceptance', vn, acc[vn])
    R['acceptance'] = acc
R['patch_clean_log'] = dict(CLEAN_LOG); json.dump(R, open(DOUT / f'gor_dense_eval{DTAG}.json', 'w'), ensure_ascii=False, indent=1, default=float)
import resource; log('done; peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
