"""Пункт 2: вложенный LOFO для калибровки порогов (VARIANTS_DECLARED.md, C0/C1/C2) + количественный сдвиг fold→all.
Вход: audit/calib/pred_*.geojson (calib_predict.py), pixgbm/pix_lofo.geojson (M_{−F} на F), стек (geo_gap/evlib).
Выход: audit/out/calib_eval.json."""
import os, sys, json, collections
GG = '/workspace/koz3_shared/bot-3/geo_gap'; A = '/workspace/koz3_shared/bot-3/audit'
sys.path.insert(0, GG); os.chdir(GG); sys.argv = ['x']
exec(open('eval_r.py').read().split('FP = stack_preds')[0])
from evlib import *
import tomllib
CL = ['finds_points', 'object_poly']; FAMS = ['Гнёздово', 'Семибратнее', 'Судьбищи']
STACK = {c: stack_preds(c) for c in CL}
mt_of = lambda grp: grp.replace('_pix', '')
def load_pix(path):
    P = {c: [(g, cf, r, grp, 'pix') for g, cf, r, grp, _ in v] for c, v in file_preds(path).items()}
    for c in CL: P.setdefault(c, [])
    return P
def variant_S(PIX, c, base):
    cov = {(p[2], mt_of(p[3])) for p in PIX[c]}
    return [p for p in base if (p[2], p[3]) not in cov] + PIX[c]
def only(P, regs): return {c: [p for p in P[c] if p[2] in regs] for c in CL}
def union(*Ps): return {c: sum((P[c] for P in Ps), []) for c in CL}
FAMREG = {f: {r for r in REGIONS if site_family(r) == f} for f in FAMS}
PIXREG = {f: {r for r in FAMREG[f] if r[:3].isdigit() and 200 <= int(r[:3]) <= 217} for f in FAMS}
LOFO = load_pix('/workspace/koz3_shared/bot-3/pixgbm/pix_lofo.geojson')
PR = {k: load_pix(f'{A}/calib/pred_{k}.geojson') for k in ['ens2', 'all'] + [f'single_{f}' for f in FAMS] + [f'ins_{f}' for f in FAMS]}
TH = tomllib.load(open('/workspace/koz3_shared/bot-3/pixgbm/thresholds_v2.toml', 'rb'))['thresholds_by_markup']
print('thresholds_v2', TH, flush=True)

def taus_of(res):
    t = {}
    for cls, v in res.items():
        for g, gv in v['groups'].items(): t.setdefault(g, {})[cls] = gv['threshold']
    return t
def region_counts(cv, taus, regs, unseen):
    """{класс: {участок: [tp, fp, n_gt]}} на участках regs при порогах taus."""
    out = {}
    for cls, c in cv.items():
        hc = M.curve_where(c, regs); per = {r: [0, 0, int(n)] for r, n in hc.gt_regions.items()}
        if hc.conf.size:
            fill = np.inf if unseen == 'drop' else -np.inf
            tau = np.array([taus.get(g, {}).get(cls, fill) for g in hc.pred_group], float)
            kept = hc.conf >= tau
            for r, h in zip(np.asarray(hc.pred_region, dtype=object)[kept], hc.hit[kept]):
                per.setdefault(r, [0, 0, 0])[0 if h else 1] += 1
        out[cls] = per
    return out
def merge(a, b):
    o = {c: dict(a.get(c, {})) for c in CL}
    for c in CL:
        for r, v in b.get(c, {}).items(): o[c][r] = v
    return o
def boot(a, b, n=1000, seed=0):
    regs = sorted({r for cs in (a, b) for by in cs.values() for r in by})
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        k = np.bincount(rng.integers(0, len(regs), len(regs)), minlength=len(regs))
        w = {r: float(x) for r, x in zip(regs, k) if x}
        d.append(M.score_counts(a, None, w)[0] - M.score_counts(b, None, w)[0])
    d = np.array(d); return dict(mean=float(d.mean()), p5=float(np.percentile(d, 5)), p95=float(np.percentile(d, 95)), p_gt0=float((d > 0).mean()))
def cnt_ge(P, c, grp, regs, t): return sum(1 for p in P[c] if p[3] == grp and p[2] in regs and p[1] >= t)
def tau_count(P, c, grp, regs, n):
    v = sorted((p[1] for p in P[c] if p[3] == grp and p[2] in regs), reverse=True)
    if n <= 0: return float('inf')
    return v[min(n, len(v)) - 1] if v else float('inf')

res = {'shift': {}, 'nested': {}}
# ── сдвиг fold → all на трейне: число кандидатов ≥ порогов thresholds_v2 (in-sample all против OOF fold) ──
thr = {}
for g, v in TH.items():
    if isinstance(v, dict):
        for cls, t in v.items():
            if cls in CL: thr[(g, cls)] = float(t)
for (g, cls), t in sorted(thr.items()):
    for f in FAMS:
        res['shift'][f'{g}|{cls}|{f}'] = dict(tau=t, n_lofo=cnt_ge(LOFO, cls, g, PIXREG[f], t), n_all_insample=cnt_ge(PR['all'], cls, g, PIXREG[f], t),
                                             n_lofo_any=cnt_ge(LOFO, cls, g, PIXREG[f], -1), n_all_any=cnt_ge(PR['all'], cls, g, PIXREG[f], -1))
    print('shift', g, cls, {f: res['shift'][f'{g}|{cls}|{f}'] for f in FAMS}, flush=True)

for scope in ['global', 'site']:
    for unseen in ['drop', 'keep']:
        tot = {k: {} for k in ['C0', 'C1', 'C2', 'oracle']}
        per_f = {}
        for F in FAMS:
            Gf, Hf = [x for x in FAMS if x != F]
            TR = FAMREG[Gf] | FAMREG[Hf]
            nested = union(only(PR[f'single_{Hf}'], PIXREG[Gf]), only(PR[f'single_{Gf}'], PIXREG[Hf]))
            cvN = curves_of({c: variant_S(nested, c, STACK[c]) for c in CL}, scope)
            tau = taus_of(M.sweep_groups({c: M.curve_where(cv, TR) for c, cv in cvN.items()}))
            pixg = sorted({p[3] for c in CL for p in nested[c]})
            ins = only(PR[f'ins_{F}'], PIXREG[Gf] | PIXREG[Hf])
            tau1 = json.loads(json.dumps(tau)); cm = {}
            for g in pixg:
                for c in CL:
                    if c in tau.get(g, {}):
                        n_oof = cnt_ge(nested, c, g, TR, tau[g][c]); t1 = tau_count(ins, c, g, TR, n_oof)
                        tau1[g][c] = t1; cm[f'{g}|{c}'] = dict(tau=tau[g][c], n_oof=n_oof, n_ins_at_tau=cnt_ge(ins, c, g, TR, tau[g][c]), tau_C1=t1)
            cvL = curves_of({c: variant_S(only(LOFO, PIXREG[F]), c, STACK[c]) for c in CL}, scope)
            cvE = curves_of({c: variant_S(only(PR['ens2'], PIXREG[F]), c, STACK[c]) for c in CL}, scope)
            rc = {'C0': region_counts(cvL, tau, FAMREG[F], unseen), 'C1': region_counts(cvL, tau1, FAMREG[F], unseen),
                  'C2': region_counts(cvE, tau, FAMREG[F], unseen),
                  'oracle': region_counts(cvL, taus_of(M.sweep_groups({c: M.curve_where(cv, FAMREG[F]) for c, cv in cvL.items()})), FAMREG[F], unseen)}
            for k in tot: tot[k] = merge(tot[k], rc[k])
            per_f[F] = dict(tau={g: tau.get(g) for g in pixg}, count_match=cm,
                            score={k: M.score_counts(v)[0] for k, v in rc.items()},
                            per_class={k: {c: {kk: M.score_counts(v)[1][c][kk] for kk in ('tp', 'fp', 'n_gt', 'f1')} for c in CL} for k, v in rc.items()})
            print(scope, unseen, F, json.dumps(per_f[F], ensure_ascii=False, default=float), flush=True)
        sc = {k: M.score_counts(v)[0] for k, v in tot.items()}
        r = dict(per_family=per_f, pooled=sc, pooled_per_class={k: M.score_counts(v)[1] for k, v in tot.items()},
                 d_C1=sc['C1'] - sc['C0'], d_C2=sc['C2'] - sc['C0'], boot_C1=boot(tot['C1'], tot['C0']), boot_C2=boot(tot['C2'], tot['C0']))
        res['nested'][f'{scope}|{unseen}'] = r
        print('POOLED', scope, unseen, json.dumps({k: r[k] for k in ('pooled', 'd_C1', 'd_C2', 'boot_C1', 'boot_C2')}, default=float), flush=True)
json.dump(res, open(f'{A}/out/calib_eval.json', 'w'), ensure_ascii=False, indent=1, default=float)
