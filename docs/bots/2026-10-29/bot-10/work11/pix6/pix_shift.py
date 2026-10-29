"""bot-10 work11 задача 6: сдвиг калибровки PIXGBM — LOFO-OOF (bot-3/pixgbm/pix_lofo.geojson, модели без своей семьи)
против all-модели на всём трейне (bot-10/in/geo_pix_S_train.geojson, in-sample!). Только существующие артефакты, без обучения/инференса.
(а) распределения по (группа, класс) на общих участках: квантили, KS, доля/число >= τ (bot-3/pixgbm/pc/thr_pix_full.json),
    квантильно-согласованный порог τq = F_full^-1(F_oof(τ)), по семьям;
(б) влияние на метрику (honest-10, кадр bot-3 eval_pix, вариант S): пороги учатся на OOF (как в сдаче), а оценки теста
    сдвинуты как all-модель in-sample → на OOF это порог τ_eff = g^-1(τ) (g — монотонное отображение OOF→full, квантильное
    или по числу). «Оставить» = τ_eff; «квантильно согласовать» = исходный τ (по построению). Δ = оставить − согласовать.
usage: python pix_shift.py   (пишет pix_shift.json в cwd)"""
import os, sys, json, collections
import numpy as np
from scipy.stats import ks_2samp
OUTD = os.getcwd()
W3 = '/workspace/koz3_shared/bot-3/pixgbm'
FULL = '/workspace/koz3_shared/bot-10/in/geo_pix_S_train.geojson'
LOFOF = f'{W3}/pix_lofo.geojson'
T = json.load(open(f'{W3}/pc/thr_pix_full.json'))
def load(p):
    C = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in json.load(open(p, encoding='utf-8'))['features']:
        q = f['properties']; C[(q.get('threshold_group'), q['class_name'])][q['region_name']].append(float(q['confidence']))
    return C
L, F = load(LOFOF), load(FULL)
fam = lambda r: 'Судьбищи' if r.startswith('200') else ('Семибратнее' if int(r[:3]) <= 207 else 'Гнёздово')
def frac_ge(v, t): return float((v >= t).mean()) if len(v) else None
def qmatch(src, dst, t):
    """порог на шкале dst с той же долей выше, что у t на шкале src"""
    p = (src >= t).mean()
    return float(np.quantile(dst, 1 - p)) if 0 < p < 1 else (float(dst.min()) if p >= 1 else float(dst.max()) + 1e-9)
def cmatch(src, dst, t):
    """порог на шкале dst с тем же ЧИСЛОМ объектов выше, что у t на шкале src"""
    n = int((src >= t).sum()); s = np.sort(dst)[::-1]
    return float(s[n - 1]) if 0 < n <= len(s) else (float(s[-1]) if n > len(s) else float(s[0]) + 1e-9)
A = {}; MAPS = {}
for key in sorted(set(L) | set(F)):
    g, c = key; tau = T.get(f'S|global|{c}', {}).get(g)
    regs = sorted(set(L[key]) & set(F[key]))
    lo = np.concatenate([L[key][r] for r in regs]) if regs else np.array([]); fu = np.concatenate([F[key][r] for r in regs]) if regs else np.array([])
    floor = max(lo.min(), fu.min()) if len(lo) and len(fu) else None
    lo2, fu2 = lo[lo >= floor], fu[fu >= floor]   # общий пол кандидатов
    qs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    r = {'regions_lofo': sorted(L[key]), 'regions_full': sorted(F[key]), 'common': regs, 'floor_lofo': float(lo.min()) if len(lo) else None,
         'floor_full': float(fu.min()) if len(fu) else None, 'common_floor': floor, 'n_lofo': int(len(lo2)), 'n_full': int(len(fu2)), 'tau': tau}
    if len(lo2) and len(fu2):
        ks = ks_2samp(fu2, lo2)
        r.update(q_lofo={q: float(np.quantile(lo2, q)) for q in qs}, q_full={q: float(np.quantile(fu2, q)) for q in qs},
                 ks_D=float(ks.statistic), ks_p=float(ks.pvalue), mean_lofo=float(lo2.mean()), mean_full=float(fu2.mean()))
        if tau is not None:
            r.update(frac_ge_tau_lofo=frac_ge(lo2, tau), frac_ge_tau_full=frac_ge(fu2, tau), n_ge_tau_lofo=int((lo2 >= tau).sum()), n_ge_tau_full=int((fu2 >= tau).sum()),
                     tau_qmatched_full=qmatch(lo2, fu2, tau), tau_cmatched_full=cmatch(lo2, fu2, tau))
            r['shift_q'] = r['tau_qmatched_full'] - tau; r['shift_c'] = r['tau_cmatched_full'] - tau
        per = {}
        for fm in sorted({fam(x) for x in regs}):
            rr = [x for x in regs if fam(x) == fm]; a = np.concatenate([L[key][x] for x in rr]); b = np.concatenate([F[key][x] for x in rr])
            a, b = a[a >= floor], b[b >= floor]
            if len(a) and len(b):
                per[fm] = {'regions': len(rr), 'n_lofo': int(len(a)), 'n_full': int(len(b)), 'median_lofo': float(np.median(a)), 'median_full': float(np.median(b)),
                           'ks_D': float(ks_2samp(b, a).statistic)}
                if tau is not None: per[fm].update(frac_ge_tau_lofo=frac_ge(a, tau), frac_ge_tau_full=frac_ge(b, tau), tau_qmatched_full=qmatch(a, b, tau))
        r['per_family'] = per
        MAPS[key] = (lo2, fu2)
    A[f'{g}|{c}'] = r
    print(g, c, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ('per_family', 'regions_lofo', 'regions_full', 'common', 'q_lofo', 'q_full')}, ensure_ascii=False), flush=True)
    for fm, v in r.get('per_family', {}).items(): print('   ', fm, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}, flush=True)
json.dump({'dist': A}, open(f'{OUTD}/pix_shift_dist.json', 'w'), ensure_ascii=False, indent=1, default=float)

# ───── (б) метрика: кадр bot-3 eval_pix (вариант S), только чтение ─────
if os.environ.get('NO_METRIC'): sys.exit(0)
os.environ['PART'] = 'none'; os.environ['PAIRS'] = 'S:stack'
sys.argv = ['eval_pix.py']
src = open(f'{W3}/eval_pix.py').read().split('# ───────── (1) адресная')[0]
exec(src)
os.chdir(OUTD)
def g_inv(key, kind):
    """τ на шкале full → эквивалентный порог на шкале OOF (что сделает неизменный τ при сдвиге как у all-модели in-sample)"""
    if key not in MAPS: return lambda t: t
    lo2, fu2 = MAPS[key]
    return (lambda t: qmatch(fu2, lo2, t)) if kind == 'q' else (lambda t: cmatch(fu2, lo2, t))
def honest_counts_shift(curves, folds, unseen, kind):
    every = set().union(*folds); counts = {cls: {} for cls in curves}; used = collections.defaultdict(list)
    for held in folds:
        res = M.sweep_groups({cls: M.curve_where(cv, every - held) for cls, cv in curves.items()})
        taus = {}
        for cls, v in res.items():
            for g, gv in v['groups'].items(): taus.setdefault(g, {})[cls] = gv['threshold']
        for cls, cv in curves.items():
            hc = M.curve_where(cv, held); per = counts[cls]
            for r, n in hc.gt_regions.items(): per.setdefault(r, [0, 0, 0])[2] += int(n)
            if hc.conf.size == 0: continue
            tt = {}
            for g in set(hc.pred_group):
                t = taus.get(g, {}).get(cls)
                if t is None: tt[g] = np.inf if unseen == 'drop' else -np.inf
                elif kind is None or not g.endswith('_pix'): tt[g] = t
                else: tt[g] = g_inv((g, cls), kind)(t); used[f'{g}|{cls}'].append((round(t, 6), round(tt[g], 6)))
            tau = np.array([tt[g] for g in hc.pred_group]); kept = hc.conf >= tau
            for r, hit in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]): per.setdefault(r, [0, 0, 0])[0 if hit else 1] += 1
    full = dict(BASE['counts']); full.update(counts)
    return full, used
out = {}
for scope in ['global', 'site']:
    cv = curves_of(VARS['S'], scope)
    for unseen in ['keep', 'drop']:
        base, _ = honest_counts_shift(cv, FOLDS, unseen, None)
        sb = scores(base)
        for kind in ['q', 'c']:
            var, used = honest_counts_shift(cv, FOLDS, unseen, kind)
            sv = scores(var)
            pb = M.paired_bootstrap_counts(var, base, site_family, n=1000)
            lofo = scores(honest_counts_shift(cv, LOFO, unseen, kind)[0])[0] - scores(honest_counts_shift(cv, LOFO, unseen, None)[0])[0]
            rnd = [scores(honest_counts_shift(cv, f, unseen, kind)[0])[0] - scores(honest_counts_shift(cv, f, unseen, None)[0])[0] for f in RAND]
            pa, pbb = M.score_counts(var)[1], M.score_counts(base)[1]
            key = f'{scope}|{unseen}|{kind}'
            out[key] = {'S_as_is_kept_tau(shifted)': sv, 'S_qmatched(=OOF)': sb, 'd_keep_minus_match': sv[0] - sb[0], 'd_new34': sv[1] - sb[1], 'p_gt0': pb.get('p_gt0'),
                        'boot': pb, 'lofo': lofo, 'rand': rnd,
                        'per_class': {c: {'base': {k: pbb[c][k] for k in ('tp', 'fp', 'n_gt', 'f1')}, 'shifted': {k: pa[c][k] for k in ('tp', 'fp', 'n_gt', 'f1')}} for c in CL},
                        'taus_used_example': {k: sorted(set(v))[:12] for k, v in used.items()}}
            print(key, 'S(OOF-τ, =квант. согласование)', [round(x, 6) for x in sb], 'S(τ как есть, сдвиг in-sample)', [round(x, 6) for x in sv],
                  'Δ %+.6f P(>0) %.3f lofo %+.6f rnd %s' % (sv[0] - sb[0], pb.get('p_gt0', 0), lofo, [round(x, 5) for x in rnd]),
                  {c: (pbb[c]['tp'], pbb[c]['fp'], '->', pa[c]['tp'], pa[c]['fp']) for c in CL}, flush=True)
            json.dump({'dist': A, 'metric': out}, open(f'{OUTD}/pix_shift.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('done', flush=True)
