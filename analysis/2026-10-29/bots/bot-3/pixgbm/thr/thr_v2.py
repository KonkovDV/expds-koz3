"""Пороги *_pix для сдачи (bot-3). Вариант S (как в eval_pix.variant_S, принят bot-10) на объединённых LOFO OOF
(pix_lofo.geojson: каждый участок предсказан моделью без своей семьи), официальный compute_metrics.py, scope global,
весь трейн 200–217 (+ остальные классы — base_honest.pkl, постоянны). Не-pix группы — пороги сдачи (sub_geo.pkl уже
отсечён min_confidence; etgr object_poly ещё 0,096635 из [thresholds_by_markup.etgr]).
Подбор: покоординатный подъём по (группа, класс) *_pix при прочих фиксированных; финал перепроверен честным
пересопоставлением (match по отфильтрованным кандидатам). Устойчивость: оптимумы без каждой семьи (LOFO-трейн),
оптимум внутри семьи, плато ±0,05."""
import os, sys, json, collections
os.environ.setdefault('OMP_NUM_THREADS', '1')
GG = '/workspace/koz3_shared/bot-3/geo_gap'; W = '/workspace/koz3_shared/bot-3/pixgbm'
sys.path.insert(0, GG); os.chdir(GG)
from evlib import *
import numpy as np
os.chdir(W + '/thr')
CL = ['finds_points', 'object_poly']
PIX = {c: [(g, cf, r, grp, 'pix') for g, cf, r, grp, _ in v] for c, v in file_preds(f'{W}/pix_lofo.geojson').items()}
for c in CL: PIX.setdefault(c, [])
mt_of = lambda grp: grp.replace('_pix', '')
STACK = {c: stack_preds(c) for c in CL}
def variant_S(c, base):
    cov = {(p[2], mt_of(p[3])) for p in PIX[c]}
    return [p for p in base if (p[2], p[3]) not in cov] + PIX[c]
SV = {c: variant_S(c, STACK[c]) for c in CL}
print('n', {c: collections.Counter(p[3] for p in SV[c]) for c in CL}, flush=True)
CV = curves_of(SV, 'global')
FIX = {('etgr', 'object_poly'): 0.096635}
PG = sorted({(p[3], c) for c in CL for p in SV[c] if p[3].endswith('_pix')})
print('pix groups', PG, flush=True)
REG_FAM = {r: site_family(r) for r in REGIONS}

def kept_counts(taus, regions=None):
    """{класс: {участок: [tp, fp, n_gt]}} при порогах taus[(группа, класс)] (иначе FIX, иначе 0)."""
    out = {}
    for c in CL:
        cv = CV[c]; g = np.array(cv.pred_group, dtype=object); reg = np.array(cv.pred_region, dtype=object)
        tau = np.array([taus.get((x, c), FIX.get((x, c), 0.0)) for x in g])
        k = cv.conf >= tau
        per = {}
        for r, n in cv.gt_regions.items():
            if regions is None or r in regions: per.setdefault(r, [0, 0, 0])[2] += int(n)
        for r, h in zip(reg[k], cv.hit[k]):
            if regions is None or r in regions: per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        out[c] = per
    return out

def f1c(per):
    tp = sum(v[0] for v in per.values()); fp = sum(v[1] for v in per.values()); n = sum(v[2] for v in per.values())
    return 2 * tp / (2 * tp + fp + (n - tp)) if n else 0.0, tp, fp, n

def total(taus, counts=None):
    cs = counts or kept_counts(taus)
    full = dict(BASE['counts']); full.update(cs)
    return M.score_counts(full)[0]

def cands(g, c):
    cv = CV[c]; m = np.array(cv.pred_group, dtype=object) == g
    u = np.unique(cv.conf[m]); return np.concatenate([[0.0], u, [np.inf]])

def class_f1_fast(c, taus, regions=None):
    cv = CV[c]; g = cv.pred_group
    tau = np.array([taus.get((x, c), FIX.get((x, c), 0.0)) for x in g])
    m = np.ones(len(g), bool) if regions is None else np.array([r in regions for r in cv.pred_region])
    k = (cv.conf >= tau) & m
    tp = int(cv.hit[k].sum()); fp = int(k.sum()) - tp
    n = cv.n_gt if regions is None else sum(v for r, v in cv.gt_regions.items() if r in regions)
    return 2 * tp / (2 * tp + fp + (n - tp)) if n else 0.0

def scan(g, c, taus, regions=None):
    xs = cands(g, c)
    # быстро: заранее посчитанное без группы g
    cv = CV[c]; gg = np.array(cv.pred_group, dtype=object)
    m = np.ones(len(gg), bool) if regions is None else np.array([r in regions for r in cv.pred_region])
    tau_o = np.array([taus.get((x, c), FIX.get((x, c), 0.0)) for x in gg])
    other = (gg != g) & m & (cv.conf >= tau_o)
    tp0 = int(cv.hit[other].sum()); n0 = int(other.sum())
    sel = (gg == g) & m
    cf, ht = cv.conf[sel], cv.hit[sel]
    o = np.argsort(-cf); cf, ht = cf[o], ht[o]
    ctp = np.concatenate([[0], np.cumsum(ht)]); cn = np.arange(len(cf) + 1)
    # при пороге x оставлены cf >= x: число = searchsorted по убыванию
    idx = np.array([np.searchsorted(-cf, -x, side='right') for x in xs])
    n_gt = cv.n_gt if regions is None else sum(v for r, v in cv.gt_regions.items() if r in regions)
    tp = tp0 + ctp[idx]; nn = n0 + cn[idx]
    f = 2 * tp / (nn + n_gt) if n_gt else np.zeros(len(xs))
    return xs, f

def optimize(regions=None, start=0.0):
    taus = {k: start for k in PG}
    for _ in range(20):
        ch = False
        for (g, c) in PG:
            xs, f = scan(g, c, taus, regions)
            j = int(np.argmax(f))   # первый максимум — наименьший порог
            cur = class_f1_fast(c, taus, regions)
            if f[j] > cur + 1e-12 and xs[j] != taus[(g, c)]:
                taus[(g, c)] = float(xs[j]); ch = True
        if not ch: break
    return taus

res = {}
default = {k: 0.0 for k in PG}   # сдача S без [thresholds_by_markup.*_pix]: [thresholds] object_poly/finds_points = 0.0
best = None
for st in (0.0, 1.0):
    t = optimize(None, st)
    s = total(t)
    if best is None or s > best[1]: best = (t, s)
taus = best[0]
cs_opt, cs_def = kept_counts(taus), kept_counts(default)
res['taus_opt'] = {f'{g}|{c}': v for (g, c), v in taus.items()}
res['score_opt_fast'] = total(taus, cs_opt); res['score_default_fast'] = total(default, cs_def)
res['class_opt'] = {c: f1c(cs_opt[c]) for c in CL}; res['class_default'] = {c: f1c(cs_def[c]) for c in CL}
# плато: F1 класса при τ ± 0,05 (остальные на оптимуме) и диапазон τ с потерей ≤ 0,002 / 0,005 F1 класса
plat = {}
for (g, c) in PG:
    xs, f = scan(g, c, taus)
    fx = lambda x: float(f[np.searchsorted(xs, x, side='right') - 1]) if np.isfinite(x) else float(f[-1])
    t0 = taus[(g, c)]; fb = float(f.max())
    fin = np.isfinite(xs)
    within = lambda tol: [float(xs[fin][f[fin] >= fb - tol].min()), float(xs[fin][f[fin] >= fb - tol].max())]
    grid = {f'{x:.2f}': fx(x) for x in np.round(np.arange(0, 1.0001, 0.05), 2)}
    plat[f'{g}|{c}'] = dict(tau=t0, f1=fb, f1_m005=fx(max(0, t0 - 0.05)), f1_p005=fx(t0 + 0.05),
                           range_tol002=within(0.002), range_tol005=within(0.005), f1_at_0=fx(0.0), grid=grid)
res['plateau'] = plat
# устойчивость: оптимум без семьи (как в LOFO-фолде) и внутри семьи
geo_fams = sorted({REG_FAM[r] for c in CL for r in CV[c].gt_regions})
fam_regs = lambda f: {r for r in REGIONS if REG_FAM[r] == f}
res['lofo_train_taus'] = {}; res['in_family_taus'] = {}
for fam in geo_fams:
    tr = REGIONS - fam_regs(fam)
    res['lofo_train_taus'][fam] = {f'{g}|{c}': v for (g, c), v in optimize(tr).items()}
    res['in_family_taus'][fam] = {f'{g}|{c}': v for (g, c), v in optimize(fam_regs(fam)).items()}
# точная перепроверка: заново сопоставить только оставленные кандидаты
def exact(t):
    P = {c: [p for p in SV[c] if p[1] >= t.get((p[3], c), FIX.get((p[3], c), 0.0))] for c in CL}
    cv = curves_of(P, 'global')
    cs = {}
    for c in CL:
        per = {r: [0, 0, int(n)] for r, n in cv[c].gt_regions.items()}
        for r, h in zip(cv[c].pred_region, cv[c].hit):
            per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        cs[c] = per
    return total(t, cs), {c: f1c(cs[c]) for c in CL}
res['exact_opt'] = exact(taus); res['exact_default'] = exact(default)
json.dump(res, open('thr_v2.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'plateau'}, ensure_ascii=False, default=float))
for k, v in plat.items(): print(k, {a: b for a, b in v.items() if a != 'grid'})
