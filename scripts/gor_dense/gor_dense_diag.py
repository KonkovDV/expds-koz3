# bot-5. ТОЛЬКО ЧЕСТНЫЙ ЗАМЕР. Диагностика просадки new34 у вариантов с точками dense и честные фиксы.
"""Переиспользует голову gor_dense_eval.py (те же peaks.json, гейт v3, sweep_groups только на обучающих участках).
MODE=diag: по раскладке fam10 для VARS (напр. all+cp:add+gate) — по участкам new34 и всем: tp/fp по происхождению
(dense / стек) для контроля (gate или base) и варианта; пороги групп; тип сцены.
MODE=fix: фиксы, параметр выбирается В КАЖДОМ внешнем фолде по F1 класса на обучающих участках (sweep_groups),
отложенное семейство не участвует: NEAR — подавить точку ближе D м к строке стека gor той же сцены
(D ∈ {none,30,100,300,scene}); PMIN — точка только при пике ≥ t (t ∈ {0,.5,.7,.9}). Env: MODE, VARS, FIXES, LAYOUTS, DOUT."""
import os, sys, json, math, collections, time
from pathlib import Path
import numpy as np
_ev = (Path(__file__).resolve().parent / 'gor_dense_eval.py').read_text(encoding='utf-8').split("\nR = {'definition'")[0]
exec(compile(_ev, 'gor_dense_eval_head', 'exec'))
MODE = os.environ.get('MODE', 'diag'); ODIR = Path(os.environ.get('ODIR', '/workspace/koz3_shared/bot-5/analysis/gor_dense_diag'))
VSEL = os.environ.get('VARS', 'all+cp:add+gate,pu_edge+cp:add+gate').split(',')
gidx = {int(i): n for n, i in enumerate(gor_rows)}
SXY = collections.defaultdict(list)
for i in gor_rows: SXY[reg[i]].append((float(T['cx'][i]), float(T['cy'][i])))
def build_ids(rows, gate_p):
    ids = list(map(int, gor_rows)); cf = list(conf[gor_rows].astype(float)); rg = list(reg[gor_rows]); gp = list(grp[gor_rows])
    if gate_p is not None:
        for n, i in enumerate(ids):
            if mk[i] == 'sp_or' and rg[n] in gate_p: cf[n] = cf[n] * gate_p[rg[n]]
    extra = {}
    for k, row in enumerate(rows):
        r, c, pr = row[:3]; nid = N0 + k; extra[nid] = pr; ids.append(nid); cf.append(c); rg.append(r); gp.append('gor_dense')
    cf = np.array(cf); o = np.argsort(-cf, kind='stable'); ids = np.array([ids[k] for k in o])
    hit = greedy_hits(list(ids), lambda i: extra[i] if i >= N0 else pairs_by.get(i, []))
    cv = ClassCurve(conf=cf[o], hit=hit, pred_region=[rg[k] for k in o], n_gt=T['n_gt'][G],
                    gt_regions=collections.Counter(T['greg'][G]), pred_group=[gp[k] for k in o])
    return cv, ids
def gate_of(train, held):
    tri = ALLIDX[np.isin(PREG, list(train))]; hei = ALLIDX[np.isin(PREG, list(held))]; gp_ = {}
    key = frozenset(PREG[tri].tolist())
    if not len(hei) and key in GCACHE: return dict(GCACHE[key])
    if len(tri):
        set_proj(tri); regs = sorted(set(PREG[tri])); fo = family_folds(regs, site_family, 5); ptr = np.zeros(len(tri))
        for h in fo:
            te = np.isin(PREG[tri], list(h))
            if te.any(): ptr[te] = pred(fit_clean(tri[~te], True)[0], tri[te])
        mdl, _ = fit_clean(tri, True); phe = pred(mdl, hei) if len(hei) else np.zeros(0)
        for idx_set, pp in ((tri, ptr), (hei, phe)):
            for r in sorted(set(PREG[idx_set]) & SCNS): gp_[r] = float(pp[PREG[idx_set] == r].max())
        if not len(hei): GCACHE[key] = dict(gp_)
    return gp_
def dmin(r, x, yy):
    s = SXY.get(r)
    return min(math.hypot(x - a, yy - b) for a, b in s) if s else np.inf
def eval_cv(cv, ids, train, held, per, detail=None):
    res = sweep_groups({G: curve_where(cv, train)}); taus = {g: gv['threshold'] for g, gv in res[G]['groups'].items()}
    m = np.fromiter((r in held for r in cv.pred_region), bool, len(cv.pred_region))
    hc = curve_where(cv, held)
    for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
    tau = np.array([taus.get(g, np.inf) for g in hc.pred_group]); kept = hc.conf >= tau
    for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    if detail is not None:
        hid = ids[m]
        for i, r, h, g, c in zip(hid[kept], np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept], np.asarray(hc.pred_group, dtype=object)[kept], hc.conf[kept]):
            detail.append((int(i), r, bool(h), g, float(c)))
    return res[G]['f1'], taus
def rows_of(fr, v, gp_):
    return [(r, 0.5 * (gp_.get(r, 0.0) + ps), circle_pairs(r, x, yy), ps, dmin(r, x, yy)) for r, pk in fr.get(v, {}).items() if r in SCNS for ps, x, yy in pk]
NEW = set(new) & regions
def sc(per, regs=None): return score_counts({**base_full, G: per}, regs)[0]
R = {'A': A, 'new34_regions_in_eval': len(NEW), 'new34_sp_or_scenes': sorted(NEW & SCNS)}
log('new34 regions', len(NEW), 'of them SpOr scenes', len(NEW & SCNS))
if MODE == 'diag':
    folds = layouts['fam10']; every = set().union(*folds); out = {}
    DET = {'base': [], 'gate': []}; DET.update({vn: [] for vn in VSEL}); PER = {k: {} for k in DET}; TAUS = {k: [] for k in DET}; DMIN = {}
    for fr, held in zip(PK['layouts']['fam10'], folds):
        train = every - held; gp_ = gate_of(train, held)
        for k in DET:
            if k == 'base': cv, ids = build_ids([], None)
            elif k == 'gate': cv, ids = build_ids([], gp_)
            else:
                v, w = k.split(':'); rw = rows_of(fr, v, gp_); cv, ids = build_ids(rw, gp_ if 'gate' in w else None)
            dl = []; _, taus = eval_cv(cv, ids, train, held, PER[k], dl); DET[k] += dl; TAUS[k].append(taus)
            if k not in ('base', 'gate'):
                for i, r, h, g, c in dl:
                    if i >= N0: DMIN.setdefault(k, []).append((h, rw[i - N0][4], rw[i - N0][3], r in NEW))
        log('fold done', len(held))
    assert PER['base'] == {r: v for r, v in base_full[G].items()} or all(PER['base'].get(r, [0,0,0]) == v for r, v in base_full[G].items() if v != [0,0,0])
    gt_of = collections.Counter(T['greg'][G])
    def scene_type(r):
        if r not in SCNS: return 'no_SpOr_scene'
        w, h = centre[r][2], centre[r][3]; big = max(w, h) > 3000
        return ('big' if big else 'small') + ('+hf' if y[sidx[r]] else '-hf')
    for vn in VSEL:
        ctl = 'gate' if 'gate' in vn else 'base'
        rep = {'control': ctl, 'new34_score': [sc(PER[ctl], NEW), sc(PER[vn], NEW)], 'all_score': [sc(PER[ctl]), sc(PER[vn])]}
        # по участкам new34 и всем
        byr = {}
        for k, lab_ in ((ctl, 'c'), (vn, 'v')):
            for i, r, h, g, c in DET[k]:
                d = byr.setdefault(r, collections.Counter()); src = 'dense' if g == 'gor_dense' else 'stack'
                d[f'{lab_}_{src}_{"tp" if h else "fp"}'] += 1
        rows = []
        for r, d in byr.items():
            dd = {k: d[k] for k in sorted(d)}; dd.update(region=r, family=site_family(r), new34=r in NEW, gt=gt_of.get(r, 0), scene=scene_type(r))
            dd['d_tp'] = d['v_dense_tp'] + d['v_stack_tp'] - d['c_stack_tp']; dd['d_fp'] = d['v_dense_fp'] + d['v_stack_fp'] - d['c_stack_fp']
            if dd['d_tp'] or dd['d_fp']: rows.append(dd)
        rep['changed_regions'] = sorted(rows, key=lambda z: (not z['new34'], z['d_tp'] - z['d_fp']))
        agg = {}
        for sub in ('new34', 'other'):
            rr = [z for z in rows if z['new34'] == (sub == 'new34')]; a = collections.Counter()
            for z in rr:
                for k in ('v_dense_tp', 'v_dense_fp', 'v_stack_tp', 'v_stack_fp', 'c_stack_tp', 'c_stack_fp', 'd_tp', 'd_fp'): a[k] += z.get(k, 0)
            agg[sub] = dict(a)
            st = collections.Counter()
            for z in rr: st[z['scene']] += 1
            agg[sub + '_scene_types'] = dict(st)
        rep['agg_changed'] = agg
        # tp/fp/gt класса на new34 целиком
        def tot(per, regs): return [sum(per.get(r, [0,0,0])[q] for r in regs) for q in range(3)]
        rep['new34_tp_fp_gt'] = {'control': tot(PER[ctl], NEW), 'variant': tot(PER[vn], NEW)}
        rep['all_tp_fp_gt'] = {'control': tot(PER[ctl], regions), 'variant': tot(PER[vn], regions)}
        # пороги групп стека: сдвиг
        sh = collections.Counter()
        for a, b in zip(TAUS[ctl], TAUS[vn]):
            for g in a:
                if g != 'gor_dense' and abs(a[g] - b.get(g, np.inf)) > 1e-12: sh[g] += 1
        rep['tau_diffs'] = [(f, g, a[g], b.get(g)) for f, (a, b) in enumerate(zip(TAUS[ctl], TAUS[vn])) for g in a if g != 'gor_dense' and abs(a[g] - b.get(g, np.inf)) > 1e-12]
        kc = {(i, r): (g, c, h) for i, r, h, g, c in DET[ctl]}; kv = {(i, r): (g, c, h) for i, r, h, g, c in DET[vn] if i < N0}
        rep['stack_rows_lost'] = [(r, *kc[(i, r)], r in NEW) for (i, r) in kc if (i, r) not in kv]
        rep['stack_rows_gained'] = [(r, *kv[(i, r)], r in NEW) for (i, r) in kv if (i, r) not in kc]
        rep['stack_rows_hit_changed'] = [(r, *kv[(i, r)], r in NEW) for (i, r) in kv if (i, r) in kc and kc[(i, r)][2] != kv[(i, r)][2]]
        rep['stack_group_tau_changed_folds'] = dict(sh); rep['dense_tau'] = [t.get('gor_dense') for t in TAUS[vn]]
        # dense fp: расстояние до ближайшей строки стека gor
        dn = DMIN.get(vn, []); bins = [0, 30, 100, 300, 1000, np.inf]; hist = {}
        for h, d, ps, isn in dn:
            b = next(f'<{bins[q+1]}' for q in range(len(bins) - 1) if d < bins[q + 1] or q == len(bins) - 2); key = ('new34' if isn else 'other') + ' ' + b
            hist.setdefault(key, [0, 0])[0 if h else 1] += 1
        rep['dense_kept_tp_fp_by_dist_to_stack_gor'] = hist
        R[vn] = rep; log(vn, 'new34', rep['new34_score'], rep['new34_tp_fp_gt'], 'agg', agg, 'tau changed', dict(sh))
    json.dump(R, open(ODIR / 'diag_fam10.json', 'w'), ensure_ascii=False, indent=1, default=float)
if MODE == 'fix':
    # Фикс F (frozen): пороги групп стека = пороги контроля (тот же гейт без точек), подобранные sweep_groups на обучающих
    # участках; порог группы gor_dense — максимум F1 класса на обучающих участках при этих порогах (равенство -> меньше точек).
    # Фикс FN (frozen+near): то же + точка удаляется, если ближе D м к любой строке стека gor той же сцены;
    # D ∈ {none, 30, 100, 300, 1000, scene} выбирается в каждом внешнем фолде по F1 обучающих участков (равенство -> больший D).
    DGRID = [None, 30, 100, 300, 1000, 'scene']
    FIXES = os.environ.get('FIXES', 'F,FN').split(','); LAY = os.environ.get('LAYOUTS', ','.join(PK['layouts'])).split(',')
    VFIX = [f'{v}:{w}' for v in VARS_D for w in ('add', 'add+gate')]
    def frozen_pick(cv, train, taus_c):
        ct = curve_where(cv, train); g = np.asarray(ct.pred_group, dtype=object); dn = g == 'gor_dense'
        tau = np.array([taus_c.get(x, np.inf) for x in g]); ks = (~dn) & (ct.conf >= tau)
        tp0, n0 = int(ct.hit[ks].sum()), int(ks.sum()); dc = ct.conf[dn]; dh = ct.hit[dn]; o = np.argsort(-dc, kind='stable')
        best_t, best_f = np.inf, float(M._f1(tp0, n0 - tp0, ct.n_gt - tp0)) if n0 else 0.0
        ctp, cn = np.cumsum(dh[o]), np.arange(1, len(o) + 1)
        for k in range(len(o)):
            if k + 1 < len(o) and dc[o[k + 1]] == dc[o[k]]: continue
            tp, n = tp0 + int(ctp[k]), n0 + int(cn[k]); f = float(M._f1(tp, n - tp, ct.n_gt - tp))
            if f > best_f + 1e-12: best_f, best_t = f, float(dc[o[k]])
        return best_t, best_f
    def held_counts(cv, held, taus_c, t_d, per):
        hc = curve_where(cv, held); g = np.asarray(hc.pred_group, dtype=object)
        tau = np.array([t_d if x == 'gor_dense' else taus_c.get(x, np.inf) for x in g]); kept = hc.conf >= tau
        for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
        for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
    def keep_D(row, D):
        if D is None: return True
        if D == 'scene': return not math.isfinite(row[4])
        return row[4] > D
    OUTF = {'definition': 'scripts/gor_dense/gor_dense_diag.py docstring (MODE=fix)', 'A_honest10': A, 'layouts': {}, 'chosen': {}}
    for nm in LAY:
        folds = layouts[nm]; every = set().union(*folds); per = {f'{fx}|{vn}': {} for fx in FIXES for vn in VFIX}; ch = collections.defaultdict(list)
        for fr, held in zip(PK['layouts'][nm], folds):
            train = every - held; gp_ = gate_of(train, held); tc = {}
            for gate in (False, True):
                cv, ids = build_ids([], gp_ if gate else None)
                res = sweep_groups({G: curve_where(cv, train)}); tc[gate] = {g: gv['threshold'] for g, gv in res[G]['groups'].items()}
            for v in VARS_D:
                rw = rows_of(fr, v, gp_)
                for w in ('add', 'add+gate'):
                    gate = 'gate' in w; vn = f'{v}:{w}'; cache = {}
                    for fx in FIXES:
                        grid = [None] if fx == 'F' else DGRID; best = None
                        for D in grid:
                            if D not in cache:
                                cv, ids = build_ids([r_ for r_ in rw if keep_D(r_, D)], gp_ if gate else None); t_d, f = frozen_pick(cv, train, tc[gate]); cache[D] = (cv, t_d, f)
                            cv, t_d, f = cache[D]
                            if best is None or f >= best[2] - 1e-12: best = (D, t_d, f, cv)
                        D, t_d, f, cv = best; held_counts(cv, held, tc[gate], t_d, per[f'{fx}|{vn}']); ch[f'{fx}|{vn}'].append([D, t_d])
        out = {}
        b = gsc(gor_counts(layouts[nm], None))
        for k, pv in per.items():
            vv = gsc(pv); o_ = dict(base=b, var=vv, delta=15 * (vv['f1'] - b['f1']) / 100)
            if nm == 'fam10':
                vf = dict(base_full); vf[G] = pv
                o_.update(delta_exact=score_counts(vf)[0] - A, delta_new34=score_counts(vf, new)[0] - score_counts(base_full, new)[0],
                          delta_optical=score_counts(vf, optical)[0] - score_counts(base_full, optical)[0],
                          bootstrap=paired_bootstrap_counts(vf, base_full, site_family, n=1000, seed=0))
            out[k] = o_
        OUTF['layouts'][nm] = out; OUTF['chosen'][nm] = dict(ch); log(nm, 'done', {k: round(v['delta'], 5) for k, v in out.items()})
        json.dump(OUTF, open(ODIR / 'fix_eval.json', 'w'), ensure_ascii=False, indent=1, default=float)
    acc = {}
    for k, d in OUTF['layouts']['fam10'].items():
        seeds = [OUTF['layouts'][f'seed{s}'][k]['delta'] for s in range(5) if f'seed{s}' in OUTF['layouts']]
        lofo = OUTF['layouts'].get('LOFO', {}).get(k, {}).get('delta')
        acc[k] = {'P1': [d['delta_exact'], d['delta_exact'] >= 0.002], 'P2': [d['bootstrap'].get('p_gt0'), (d['bootstrap'].get('p_gt0') or 0) >= 0.85],
                  'P3': [lofo, seeds, lofo is not None and lofo > 0 and len(seeds) == 5 and sum(x > 0 for x in seeds) >= 4],
                  'P4': [d['delta_new34'], d['delta_optical'], d['delta_exact'] >= -0.0005 and d['delta_new34'] >= -0.001 and d['delta_optical'] >= -0.001],
                  'P5_below_MDE': d['delta_exact'] < 0.006}
        log('acceptance', k, acc[k])
    OUTF['acceptance'] = acc; json.dump(OUTF, open(ODIR / 'fix_eval.json', 'w'), ensure_ascii=False, indent=1, default=float)
import resource; log('done; peak RSS MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
