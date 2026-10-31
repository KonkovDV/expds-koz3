"""THR-OFF (bot-4): выключение пар «модальность × класс» (τ = +∞) — вложенный выбор (OFFn) и фиксированные диагностики.
Модальности: Or = markup or+sp_or, Ae = markup ae (как в GAP.md). Запуск: OMP_NUM_THREADS=1 nice -n 10 venv/bin/python work/thr/thr_off.py --layouts canon"""
import sys
_argv = sys.argv[1:]; sys.argv = [sys.argv[0], '--layouts', 'none']
exec(open(Path(__file__).parent / 'thr_eval.py' if False else '/workspace/koz3_shared/bot-4/work/thr/thr_eval.py').read().split('t0 = time.time(); res =')[0])
from expds.metric import ClassCurve
ap2 = argparse.ArgumentParser(); ap2.add_argument('--layouts', default='canon'); ap2.add_argument('--boot', type=int, default=1000); a2 = ap2.parse_args(_argv)
mk_of = {}
for c in range(len(CL)):
    m = cls == c
    for g, k in set(zip(T['grp'][m].tolist(), T['mk'][m].tolist())): mk_of.setdefault(g, set()).add(k)
MOD = {g: ('Or' if ks & {'or', 'sp_or'} else 'Ae' if 'ae' in ks else 'other') for g, ks in mk_of.items()}
print({g: (sorted(ks), MOD[g]) for g, ks in mk_of.items()}, flush=True)
PAIRS = [('kurgany_povrezhdennye', 'Or'), ('kurgany_tselye', 'Or'), ('gorodishcha', 'Or'), ('arkhitektury', 'Or'),
         ('kurgany_povrezhdennye', 'Ae'), ('kurgany_tselye', 'Ae'), ('fortifikatsii', 'Ae'), ('gorodishcha', 'Ae'), ('arkhitektury', 'Ae')]
FIXED = {'EXp': {('kurgany_povrezhdennye', 'Ae'), ('kurgany_tselye', 'Ae'), ('gorodishcha', 'Ae'), ('arkhitektury', 'Ae')} | {('or_only', 'nonarch')},
         'HT': {('or_markup', 'all')},
         'OrBroad': {('kurgany_povrezhdennye', 'Or'), ('kurgany_tselye', 'Or'), ('gorodishcha', 'Or')}}
def off_mask(cv_c, c, off):
    """True = кандидат остаётся. off: множество (класс, модальность) или спец-ключи."""
    g = np.array(cv_c.pred_group, dtype=object); keep = np.ones(len(g), bool)
    for (cc, mm) in off:
        if cc == 'or_markup': keep &= np.array([mk_of.get(x) != {'or'} for x in g])
        elif cc == 'or_only': keep &= np.array([not (mk_of.get(x) == {'or'} and c != 'arkhitektury') for x in g])
        elif cc == c: keep &= np.array([MOD[x] != mm for x in g])
    return keep
def sub(cv_c, keep):
    if keep.all(): return cv_c
    pr = np.asarray(cv_c.pred_region, dtype=object); pg = np.asarray(cv_c.pred_group, dtype=object)
    return ClassCurve(conf=cv_c.conf[keep], hit=cv_c.hit[keep], pred_region=pr[keep].tolist(), n_gt=cv_c.n_gt,
                      gt_regions=cv_c.gt_regions, pred_group=pg[keep].tolist())
def apply_off(cv, off): return {c: sub(v, off_mask(v, c, off)) for c, v in cv.items()}
def fold_counts(counts, cv, train, held, off):
    cvo = apply_off(cv, off); tb = taus(cvo, train); add(counts, cvo, held, {'S0': (tb, None)}, {})
def nested_off(cv, train, inner_folds):
    chosen = set()
    for (c, mm) in PAIRS:
        one = {c: cv[c]}; cnt0 = {c: {}}; cnt1 = {c: {}}
        if not (~off_mask(cv[c], c, {(c, mm)})).any(): continue
        for ih in inner_folds:
            ih = ih & train
            if not ih: continue
            it = train - ih
            for off, cn in ((set(), cnt0), ({(c, mm)}, cnt1)):
                o = apply_off(one, off); add(cn, o, ih, {'S0': (taus(o, it), None)}, {})
        f0 = score_counts(cnt0)[1][c]['f1']; f1 = score_counts(cnt1)[1][c]['f1']
        if f1 > f0: chosen.add((c, mm))
    return chosen
def run(layout):
    folds = canon if layout == 'canon' else (family_folds(regions, site_family, None) if layout == 'lofo' else rff(regions, 10, int(layout[4:])))
    outer, inner_of = load_rr(layout); fold_of = {r: i for i, f in enumerate(folds) for r in f}; fid = np.array([fold_of[r] for r in reg])
    V = ['A', 'R', 'A_OFFn', 'R_OFFn'] + [f'A_{k}' for k in FIXED] + [f'R_{k}' for k in FIXED]
    counts = {v: {c: {} for c in CL} for v in V}; ch = {'A_OFFn': [], 'R_OFFn': []}; every = set().union(*folds)
    for k, held in enumerate(folds):
        train = every - held
        inner_folds = [f - held for f in canon if f - held] if layout == 'lofo' else [f for j, f in enumerate(folds) if j != k]
        cvR = dict(base_cv); cvR['arkhitektury'] = arch_curve(mixed(outer, inner_of(k), fid != k))
        for b, cv in (('A', base_cv), ('R', cvR)):
            fold_counts(counts[b], cv, train, held, set())
            for nm, off in FIXED.items(): fold_counts(counts[f'{b}_{nm}'], cv, train, held, off)
            sel = nested_off(cv, train, inner_folds); ch[f'{b}_OFFn'].append(sorted(sel)); fold_counts(counts[f'{b}_OFFn'], cv, train, held, sel)
        if layout == 'lofo' and k % 10 == 0: print('lofo', k, f'{time.time()-t0:.0f}s', flush=True)
    return counts, ch
t0 = time.time(); res = {}
for lay in a2.layouts.split(','):
    cnt, ch = run(lay); L = {'chosen': ch}
    for v in cnt:
        ref = cnt['R'] if v.startswith('R') else cnt['A']; L[v] = summary(cnt[v], ref)
        if lay == 'canon' and v not in ('A', 'R'): L[v]['boot'] = paired_bootstrap_counts(cnt[v], ref, site_family, a2.boot)
    if lay == 'canon': pickle.dump(cnt, open(OUT / 'counts_off_canon.pkl', 'wb'))
    res[lay] = L
    print(lay, {v: (round(L[v]['score'][0], 6), round(L[v]['d'], 5)) for v in cnt}, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(OUT / f'res_off_{a2.layouts.replace(",", "_")}.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done', time.time() - t0)
